"""Core evaluation metrics for CAST: stress F1, WER, contrastive group scores."""
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

import evaluate as hf_evaluate
import jiwer
import numpy as np
import soundfile as sf
from datasets import load_dataset, Dataset
from tqdm import tqdm
from whisper_normalizer.english import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()

f1_metric = hf_evaluate.load("f1")
precision_metric = hf_evaluate.load("precision")
recall_metric = hf_evaluate.load("recall")
wer_metric = hf_evaluate.load("wer")
cer_metric = hf_evaluate.load("cer")


# ---------------------------------------------------------------------------
# Item-level helpers
# ---------------------------------------------------------------------------

def get_binary_and_words_from_item(item: dict) -> tuple:
    """Returns (binary_mask, [stressed_word]) for a dataset item."""
    sentence = normalizer(item.get("sentence", ""))
    words = sentence.split()
    binary = [0] * len(words)
    stressed_word = item.get("stressed_word", "")
    try:
        index = words.index(normalizer(stressed_word))
    except ValueError:
        try:
            index = item.get("sentence", "").split().index(stressed_word)
        except (ValueError, AttributeError):
            index = 0
    if 0 <= index < len(binary):
        binary[index] = 1
    return binary, [stressed_word]


def read_prefix_words(prefix_path: Path) -> list:
    """Read predicted stressed words from a .prefix sidecar file."""
    if not prefix_path.exists():
        return []
    s = prefix_path.read_text(encoding="utf-8").strip()
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, str):
            return [obj]
        if isinstance(obj, list):
            return [str(x) for x in obj]
        if isinstance(obj, dict):
            for k in ("stressed_word", "stressed_words", "words", "predicted_words"):
                if k in obj:
                    v = obj[k]
                    return [v] if isinstance(v, str) else [str(x) for x in v]
    except Exception:
        pass
    m = re.search(r"stressed_word\s*:\s*(.+)", s, flags=re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    return [p.strip() for p in re.split(r"[,\s]+", s) if p.strip()]


# ---------------------------------------------------------------------------
# Stress metrics
# ---------------------------------------------------------------------------

def align_and_score_stress(ref_text, pred_text, ref_stress_indices: list, pred_stress_indices: list):
    """
    Align ref/pred transcriptions with jiwer, then map stress labels through the alignment.
    Tolerates ASR substitution errors.
    """
    def to_words(t):
        if isinstance(t, str):
            return t.split()
        if isinstance(t, list):
            if len(t) == 1 and " " in t[0]:
                return t[0].split()
            return [str(w) for w in t]
        return str(t).split()

    ref_words = to_words(ref_text)
    pred_words = to_words(pred_text)
    ref_mask = [1 if i in ref_stress_indices else 0 for i in range(len(ref_words))]
    pred_mask = [1 if i in pred_stress_indices else 0 for i in range(len(pred_words))]

    output = jiwer.process_words(" ".join(ref_words), " ".join(pred_words))
    aligned_ref, aligned_pred = [], []

    for chunk in output.alignments[0]:
        r_s, r_e = chunk.ref_start_idx, chunk.ref_end_idx
        h_s, h_e = chunk.hyp_start_idx, chunk.hyp_end_idx
        if chunk.type in ("equal", "substitute"):
            for r_idx, h_idx in zip(range(r_s, r_e), range(h_s, h_e)):
                aligned_ref.append(ref_mask[r_idx])
                aligned_pred.append(pred_mask[h_idx])
        elif chunk.type == "delete":
            for r_idx in range(r_s, r_e):
                aligned_ref.append(ref_mask[r_idx])
                aligned_pred.append(0)
        elif chunk.type == "insert":
            for h_idx in range(h_s, h_e):
                aligned_ref.append(0)
                aligned_pred.append(pred_mask[h_idx])

    return aligned_ref, aligned_pred


def calc_stress_metrics(gt_item: dict, stress_data=None, y_pred=None) -> tuple:
    """
    Compute (f1, precision, recall) for a single item.

    Pass either stress_data (WhiStress output dict) or y_pred (binary list).
    """
    ref_text = normalizer(gt_item.get("sentence", "").replace("*", ""))
    ref_indices = [i for i, v in enumerate(get_binary_and_words_from_item(gt_item)[0]) if v]

    if stress_data is not None:
        assert y_pred is None
        pred_text = normalizer(" ".join([p[0] for p in stress_data.get("pairs", [])]))
        pred_indices = [i for i, p in enumerate(stress_data.get("binary", [])) if p == 1]
        y_true, y_pred = align_and_score_stress(ref_text, pred_text, ref_indices, pred_indices)
    else:
        assert y_pred is not None
        y_true = [0] * len(ref_text.split())
        if ref_indices:
            y_true[ref_indices[0]] = 1

    f1 = f1_metric.compute(predictions=y_pred, references=y_true)['f1']
    recall = precision_metric.compute(predictions=y_pred, references=y_true)['precision']
    precision = precision_metric.compute(predictions=y_pred, references=y_true)['precision']
    return f1, precision, recall


# ---------------------------------------------------------------------------
# Group-level contrastive scoring
# ---------------------------------------------------------------------------

def compute_group_scores(
    all_results: list,
    gt_key: str = 'ground_truth_stressed_words',
    pred_key: str = 'predicted_stressed_words',
    use_alignment: bool = True,
) -> tuple:
    """
    Compute pairwise contrastive stress metrics over grouped results.

    Metrics returned: (Correct, Contrast, Hit, stats)

    use_alignment=True: ASR-alignment-tolerant (for audio/WhiStress predictions).
    use_alignment=False: normalized string match (for text LLM predictions).
    """
    def _first(x):
        return x[0] if x else None

    def _norm_set(words):
        return {normalizer(w) for w in (words or [])}

    def _find_word_idx(word, words):
        nw = normalizer(word)
        for i, w in enumerate(words):
            if normalizer(w) == nw:
                return i
        return None

    def _is_stressed_string(ref_word, pred_stressed_words):
        return normalizer(ref_word) in _norm_set(pred_stressed_words)

    def _is_stressed_aligned(ref_sentence, transcription, ref_word, pred_stressed_words):
        if not transcription:
            return _is_stressed_string(ref_word, pred_stressed_words)
        ref_words = normalizer(ref_sentence.replace("*", "")).split()
        pred_words = normalizer(transcription).split()
        ref_idx = _find_word_idx(ref_word, ref_words)
        if ref_idx is None:
            return _is_stressed_string(ref_word, pred_stressed_words)
        pred_indices = [
            idx for w in (pred_stressed_words or [])
            for idx in ([_find_word_idx(w, pred_words)] if _find_word_idx(w, pred_words) is not None else [])
        ]
        try:
            y_true, y_pred = align_and_score_stress(
                " ".join(ref_words), " ".join(pred_words), [ref_idx], pred_indices
            )
            return any(yt == 1 and yp == 1 for yt, yp in zip(y_true, y_pred))
        except Exception:
            return _is_stressed_string(ref_word, pred_stressed_words)

    def _is_stressed(ref_sentence, transcription, ref_word, pred_stressed_words):
        if use_alignment:
            return _is_stressed_aligned(ref_sentence, transcription, ref_word, pred_stressed_words)
        return _is_stressed_string(ref_word, pred_stressed_words)

    groups = defaultdict(list)
    for r in all_results:
        groups[r["gt_sentence"]].append(r)

    counter_x, counter_y, counter_z = [], [], []
    bad_sizes = {"lt2": 0, "gt2": 0, "missing_gt": 0}

    for sent, items in groups.items():
        if len(items) < 2:
            bad_sizes["lt2"] += 1
            continue
        if len(items) > 2:
            bad_sizes["gt2"] += 1
            items = items[:2]

        A, B = items[0], items[1]
        gtA = _first(A.get(gt_key))
        gtB = _first(B.get(gt_key))
        if gtA is None or gtB is None:
            bad_sizes["missing_gt"] += 1
            continue

        predA = A.get(pred_key) or []
        predB = B.get(pred_key) or []
        transA = A.get("transcription", "")
        transB = B.get("transcription", "")

        gtA_in_A = _is_stressed(sent, transA, gtA, predA)
        gtB_in_A = _is_stressed(sent, transA, gtB, predA)
        gtB_in_B = _is_stressed(sent, transB, gtB, predB)
        gtA_in_B = _is_stressed(sent, transB, gtA, predB)

        Good_A = gtA_in_A and not gtB_in_A
        Good_B = gtB_in_B and not gtA_in_B

        counter_x += [int(Good_A), int(Good_B)]
        counter_y.append(int(Good_A and Good_B))
        counter_z += [gtA_in_A, gtB_in_B]

    stats = {
        "num_groups_total": len(groups),
        "num_groups_used": len(counter_y),
        "num_samples_used_in_X": len(counter_x),
        "skipped_lt2": bad_sizes["lt2"],
        "skipped_gt2": bad_sizes["gt2"],
        "skipped_missing_gt": bad_sizes["missing_gt"],
    }
    return (
        mean(counter_y) if counter_y else 0.0,
        mean(counter_x) if counter_x else 0.0,
        mean(counter_z) if counter_z else 0.0,
        stats,
    )


# ---------------------------------------------------------------------------
# Full audio evaluation loop
# ---------------------------------------------------------------------------

def _clean_stem(p) -> str:
    """Strip trailing numeric suffix added by multi-take generators (e.g. foo_00 → foo)."""
    s = Path(p).stem
    if '_' in s and s.split('_')[-1].isdigit():
        return '_'.join(s.split('_')[:-1])
    return s


def _load_audio_dataset(audio_dir) -> tuple:
    """Return (audio_files, audio_dir_path) from a directory or list of paths."""
    if isinstance(audio_dir, list):
        audio_files = audio_dir
        audio_dir = Path(audio_files[0]).parent
    else:
        audio_files = list(audio_dir.glob("*.wav"))
    return audio_files, Path(audio_dir)


def _build_gt_lookup(infer_ds) -> tuple:
    """Build filename→item, expected_files, sentence→gt_words, file→gt_word maps."""
    filename_to_item = {}
    expected_files = set()
    sentence_to_gt_words = {}
    gt_word_by_file = {}

    for item in infer_ds:
        base = Path(item.get('file_name', '')).stem or item.get('item_id', '')
        filename_to_item[base] = item
        expected_files.add(base)
        sentence = item.get("sentence")
        try:
            _, gt_words = get_binary_and_words_from_item(item)
            gt_word = gt_words[0] if gt_words else None
        except Exception:
            gt_word = None
        if gt_word and sentence:
            gt_word_norm = normalizer(gt_word)
            gt_word_by_file[base] = gt_word_norm
            sentence_to_gt_words.setdefault(sentence, set()).add(gt_word_norm)
        else:
            gt_word_by_file[base] = None

    return filename_to_item, expected_files, sentence_to_gt_words, gt_word_by_file


def _transcribe_to_sidecar(audio_files: list, dataset, whisper) -> None:
    """Run Whisper on each file; write .trans sidecar (skips if already exists)."""
    print("Transcribing with Whisper...")
    for batch in tqdm(dataset):
        audio_path = Path(batch['audio']['path'])
        trans_file = audio_path.with_suffix('.trans')
        if trans_file.exists():
            continue
        out = whisper(batch['audio']['array'], return_timestamps=True)
        trans_file.write_text(out['text'].strip(), encoding='utf-8')


def _run_whistress_on_file(
    audio_file: Path,
    transcription: str,
    gt_item: dict,
    gt_lookup: tuple,
    whistress,
) -> dict:
    """Run WhiStress on one file (reads/writes .stress sidecar). Returns per-file result dict or {}."""
    filename_to_item, _, sentence_to_gt_words, gt_word_by_file = gt_lookup
    base_clean = _clean_stem(audio_file)

    # Prefix eval
    prefix_file = audio_file.with_suffix(".prefix")
    if not prefix_file.exists():
        prefix_file = audio_file.with_name(base_clean).with_suffix(".prefix")
    prefix_words = read_prefix_words(prefix_file)

    prefix_result: dict = {}
    if gt_item and prefix_words:
        ref_text = normalizer(gt_item.get("sentence", "").replace("*", ""))
        ref_indices = [i for i, v in enumerate(get_binary_and_words_from_item(gt_item)[0]) if v]
        pred_tokens = [normalizer(w) for w in prefix_words]
        y_true_p, y_pred_p = align_and_score_stress(
            ref_text, " ".join(pred_tokens), ref_indices, list(range(len(pred_tokens)))
        )
        f1_p = f1_metric.compute(predictions=y_pred_p, references=y_true_p)["f1"]
        p_p = precision_metric.compute(predictions=y_pred_p, references=y_true_p)["precision"]
        sent = gt_item.get("sentence")
        gt_word_norm = gt_word_by_file.get(base_clean)
        pred_set = {normalizer(w) for w in prefix_words}
        other_gt = set(sentence_to_gt_words.get(sent, set()))
        if gt_word_norm:
            other_gt.discard(gt_word_norm)
        prefix_result = {
            "f1": f1_p, "precision": p_p,
            "self_hit": int(bool(gt_word_norm and (gt_word_norm in pred_set) and pred_set.isdisjoint(other_gt))),
        }

    # WhiStress eval
    stress_file = audio_file.with_suffix('.stress')
    stress_data = None
    if stress_file.exists():
        try:
            stress_data = json.loads(stress_file.read_text(encoding='utf-8'))
            # discard cached result if WhiStress produced no pairs (stale/failed run)
            if not stress_data.get('pairs'):
                stress_data = None
        except Exception:
            stress_data = None
    if stress_data is not None:
        predicted_stressed_words = stress_data.get('stressed_words', [])
        ground_truth_stressed = stress_data.get('ground_truth_stressed')
        ground_truth_stressed_words = stress_data.get('ground_truth_stressed_words')
    else:
        wav, sr = sf.read(str(audio_file))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        try:
            pairs = whistress.predict(
                audio={"sampling_rate": sr, "array": wav},
                transcription=transcription,
                return_pairs=True,
            )
            predicted_stressed_binary = [p[1] for p in pairs]
            predicted_stressed_words = [p[0] for p, b in zip(pairs, predicted_stressed_binary) if b]
        except Exception as e:
            print(f"WhiStress failed for {audio_file.name}: {e}")
            pairs = predicted_stressed_binary = predicted_stressed_words = []

        stress_data = {
            'pairs': pairs,
            'binary': predicted_stressed_binary,
            'stressed_words': predicted_stressed_words,
        }
        ground_truth_stressed = ground_truth_stressed_words = None
        if gt_item:
            ground_truth_stressed, ground_truth_stressed_words = get_binary_and_words_from_item(gt_item)
            stress_data['ground_truth_stressed'] = ground_truth_stressed
            stress_data['ground_truth_stressed_words'] = ground_truth_stressed_words
        stress_file.write_text(json.dumps(stress_data, indent=2), encoding='utf-8')

    if not (ground_truth_stressed and len(ground_truth_stressed) > 0 and gt_item):
        return {}

    ref_text = normalizer(gt_item.get("sentence", "").replace("*", ""))
    ref_index = stress_data.get('ground_truth_stressed', []).index(1) if 1 in stress_data.get('ground_truth_stressed', []) else 0
    pred_text = normalizer(" ".join([p[0] for p in stress_data.get("pairs", [])]))
    pred_indices = [i for i, b in enumerate(stress_data.get("binary", [])) if b == 1]
    y_true, y_pred = align_and_score_stress(ref_text, pred_text, [ref_index], pred_indices)
    f1 = f1_metric.compute(predictions=y_pred, references=y_true)['f1']
    precision = precision_metric.compute(predictions=y_pred, references=y_true)['precision']
    recall = precision_metric.compute(predictions=y_pred, references=y_true)['precision']

    sent = gt_item.get("sentence")
    gt_word_norm = gt_word_by_file.get(base_clean)
    pred_set = {normalizer(w) for w in predicted_stressed_words}
    other_gt = set(sentence_to_gt_words.get(sent, set()))
    if gt_word_norm:
        other_gt.discard(gt_word_norm)

    return {
        "audio_file": audio_file.name,
        "gt_sentence": gt_item['sentence'],
        "transcription": transcription,
        "ground_truth_stressed_words": ground_truth_stressed_words,
        "predicted_stressed_words": predicted_stressed_words,
        "predicted_prefix": [normalizer(w) for w in prefix_words],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "self_hit": int(bool(gt_word_norm and (gt_word_norm in pred_set) and pred_set.isdisjoint(other_gt))),
        "f1_hit": int(abs(f1 - 1.0) < 1e-9),
        "prefix_result": prefix_result,
    }


def _aggregate_wer(audio_files: list, filename_to_item: dict) -> tuple:
    """Read .trans sidecars and compute WER/CER vs ground truth. Returns (wer_scores, cer_scores)."""
    print("Computing WER...")
    wer_scores, cer_scores = [], []
    for audio_file in audio_files:
        trans_file = audio_file.with_suffix('.trans')
        if not trans_file.exists():
            continue
        predicted_text = trans_file.read_text(encoding='utf-8').strip()
        base_clean = _clean_stem(audio_file)
        gt_item = filename_to_item.get(base_clean) or filename_to_item.get(audio_file.stem)
        if gt_item:
            gt_text = (gt_item.get('text') or gt_item.get('sentence') or '').replace('**', '')
            if gt_text and predicted_text:
                wer_scores.append(wer_metric.compute(
                    predictions=[normalizer(predicted_text)], references=[normalizer(gt_text)]
                ))
                cer_scores.append(cer_metric.compute(
                    predictions=[normalizer(predicted_text)], references=[normalizer(gt_text)]
                ))
    return wer_scores, cer_scores


def evaluate_audio_metrics(
    infer_ds,
    audio_dir,
    whistress,
    whisper,
    calc_wer: bool = True,
    calc_stress: bool = True,
) -> dict:
    """Compute WER and stress metrics for generated audio files; writes .trans/.stress sidecars."""
    print("Calculating metrics...")
    audio_files, audio_dir = _load_audio_dataset(audio_dir)
    if not audio_files:
        return {}

    filename_to_item, expected_files, sentence_to_gt_words, gt_word_by_file = _build_gt_lookup(infer_ds)
    gt_lookup = (filename_to_item, expected_files, sentence_to_gt_words, gt_word_by_file)

    actual_clean = {_clean_stem(f) for f in audio_files}
    missing_samples = expected_files - actual_clean
    if missing_samples:
        print(f"WARNING: {len(missing_samples)} samples missing from audio_dir")

    audio_files = [f for f in audio_files if _clean_stem(f) in expected_files]
    total_expected = len(infer_ds)
    assert total_expected > 0, "infer_ds is empty!"
    print(f"Total expected: {total_expected} | Audio found: {len(audio_files)}")

    try:
        dataset = load_dataset("audiofolder", data_files={"train": [str(x) for x in audio_files]}, split="train")
    except Exception:
        dataset = Dataset.from_dict({"audio": [{"path": str(p), "bytes": None} for p in audio_files]})

    if calc_wer:
        _transcribe_to_sidecar(audio_files, dataset, whisper)

    all_results = []
    total_f1 = 0.0
    num_processed = 0

    if calc_stress:
        print("Predicting stress with WhiStress...")
        for audio_file in tqdm(audio_files):
            base_clean = _clean_stem(audio_file)
            trans_file = audio_file.with_suffix('.trans')
            if not trans_file.exists():
                print(f"Warning: no .trans for {audio_file.name}")
                continue
            transcription = trans_file.read_text(encoding='utf-8').strip()
            gt_item = filename_to_item.get(base_clean) or filename_to_item.get(audio_file.stem)

            result = _run_whistress_on_file(audio_file, transcription, gt_item, gt_lookup, whistress)
            if not result:
                continue

            total_f1 += result["f1"]
            num_processed += 1
            all_results.append(result)

    Correct, Contrast, Hit, stats = compute_group_scores(all_results, use_alignment=True)
    prefix_Correct, prefix_Contrast, prefix_Hit, _ = compute_group_scores(
        all_results, pred_key='predicted_prefix', use_alignment=False
    )

    wer_scores, cer_scores = [], []
    if calc_wer:
        wer_scores, cer_scores = _aggregate_wer(audio_files, filename_to_item)

    metrics = {
        "total_expected_samples": total_expected,
        "file_count": len(audio_files),
        "missing_samples": len(missing_samples),
        "avg_wer": float(np.mean(wer_scores)) if wer_scores else None,
        "avg_cer": float(np.mean(cer_scores)) if cer_scores else None,
        "Correct": Correct,
        "Contrast": Contrast,
        "Hit": Hit,
        "prefix_Correct": prefix_Correct,
        "prefix_Contrast": prefix_Contrast,
        "prefix_Hit": prefix_Hit,
        "num_processed_stress": num_processed,
        "avg_f1": total_f1 / num_processed if num_processed > 0 else None,
        "normalized_f1": total_f1 / total_expected,
    }

    results_file = audio_dir / "evaluation_results.json"
    results_file.write_text(
        json.dumps({"summary": metrics, "detailed_results": all_results, "missing_samples": list(missing_samples)}, indent=2),
        encoding="utf-8",
    )

    print("\nEvaluation Summary:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics
