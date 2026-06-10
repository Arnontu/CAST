#!/usr/bin/env python3
"""Synthesize speech for CAST items using OpenAI TTS with WhiStress quality filtering."""

import argparse
import re
import shutil
import json
from pathlib import Path
from typing import Callable, List, Dict, Any

import numpy as np
import soundfile as sf
import jsonlines
from tqdm import tqdm

import evaluate as hf_evaluate
f1_metric = hf_evaluate.load("f1")
precision_metric = hf_evaluate.load("precision")
recall_metric = hf_evaluate.load("recall")
accuracy_metric = hf_evaluate.load("accuracy")

from whistress import WhiStressInferenceClient
from openai import OpenAI
import torch
import torchaudio

TARGET_SR = 16_000
_WORD_RE = re.compile(r"[A-Za-z']+")

# ----------------------------- Utils -----------------------------

def normalize_word(w: str) -> str:
    return re.sub(r"[^a-z]+", "", (w or "").lower())

def normalize_tokens(text: str) -> List[str]:
    return [normalize_word(t) for t in re.findall(r"[A-Za-z']+", text)]

def ensure_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        if x.shape[0] == 1: x = x[0]
        elif x.shape[1] == 1: x = x[:, 0]
        else: x = x.mean(axis=0)
    elif x.ndim == 0:
        x = np.array([x], dtype=np.float32)
    return x

def word_matches(text: str):
    return list(_WORD_RE.finditer(text))

def as_int(x):
    try: return int(x)
    except: return None

def to_single_asterisk_markups(text: str) -> str:
    text = re.sub(r"<\s*strong\s*>(.*?)<\s*/\s*strong\s*>", r"*\1*", text, flags=re.IGNORECASE)
    text = re.sub(r"\*\*(.*?)\*\*", r"*\1*", text)
    return text

def _resample_to_16k_inplace(path: Path):
    """Load path, resample if needed, write back to same path at 16 kHz."""
    audio, sr = sf.read(str(path))
    audio = ensure_1d(audio)
    if sr != TARGET_SR:
        t = torch.from_numpy(audio).unsqueeze(0)
        y = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)(t).squeeze(0).numpy().astype(np.float32)
        sf.write(str(path), y, TARGET_SR, subtype="PCM_16")

# ----------------------------- Guidance -----------------------------

GUIDANCE_TEMPLATES = [
    "Read naturally. Emphasize \"{word}\" clearly.",
    "Read the sentence exactly as written. Do not change any words. Set overall speaking rate to about 95 percent of normal. On the word \"{word}\", slow to about 80 percent for that word only and raise pitch by about 160 percents. Keep all other words slightly flatter.",
    "Read the sentence exactly as written. Do not change any words. Set overall speaking rate to about 92 percent of normal. On the word \"{word}\", slow to about 76 percent for that word only and raise pitch by about 200 percents.",
    "Read the sentence exactly as written. Do not change any words. Set overall speaking rate to about 98 percent of normal. On the word \"{word}\", slow to about 82 percent for that word only and raise pitch by about 140 percents.",
]

def build_instruction(stressed_word: str, try_idx: int) -> str:
    tmpl = GUIDANCE_TEMPLATES[try_idx % len(GUIDANCE_TEMPLATES)]
    return tmpl.format(word=stressed_word)

# ----------------------------- TTS Backends -----------------------------

def make_openai_synthesize_fn(model: str = "gpt-4o-mini-tts", voice: str = "alloy") -> Callable:
    """
    Returns a synthesize_fn(text, instruction, out_path) for OpenAI TTS.

    To use a different TTS backend, implement a function with the same signature:
        synthesize_fn(text: str, instruction: str, out_path: Path) -> None
    The function should write a WAV file to out_path at any sample rate;
    it will be resampled to 16 kHz automatically.
    """
    client = OpenAI()

    def synthesize_fn(text: str, instruction: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice,
                input=text,
                instructions=instruction,
                response_format="wav",
            ) as resp:
                resp.stream_to_file(str(out_path))
        except TypeError:
            # Fallback if 'instructions' kwarg is not supported by the library version
            with client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice,
                input=text,
                response_format="wav",
            ) as resp:
                resp.stream_to_file(str(out_path))
        _resample_to_16k_inplace(out_path)

    return synthesize_fn

# ----------------------------- WhiStress Eval -----------------------------

def run_whistress(whistress, wav_path: Path, sentence_clean: str, ref_mask: List[int]) -> Dict[str, Any]:
    audio, sr = sf.read(str(wav_path))
    audio = ensure_1d(audio)

    try:
        pairs = whistress.predict(
            audio={"sampling_rate": TARGET_SR, "array": audio},
            transcription=sentence_clean,
            return_pairs=True
        )
    except Exception as e:
        return {"error": str(e), "f1": -1, "transcript_match": False}

    pred_words = [str(x[0]) for x in pairs] if pairs else []
    y_pred = [int(x[1]) for x in pairs][:len(ref_mask)] if pairs else []

    if len(y_pred) < len(ref_mask):
        y_pred += [0] * (len(ref_mask) - len(y_pred))

    try:
        p = precision_metric.compute(predictions=y_pred, references=ref_mask)["precision"]
        r = recall_metric.compute(predictions=y_pred, references=ref_mask)["recall"]
        f = f1_metric.compute(predictions=y_pred, references=ref_mask)["f1"]
        a = accuracy_metric.compute(predictions=y_pred, references=ref_mask)["accuracy"]
    except Exception:
        p = r = f = a = -1

    sent_tokens = normalize_tokens(sentence_clean)
    pred_tokens = [normalize_word(w) for w in pred_words]
    transcript_match = (pred_tokens == sent_tokens)

    failure_type = None
    if not transcript_match:
        failure_type = "Transcript"
    elif f < 1.0:
        failure_type = "Stress"

    return {
        "metrics": {"precision": float(p), "recall": float(r), "f1": float(f), "accuracy": float(a)},
        "pred_mask": y_pred,
        "pred_words": pred_words,
        "transcript_match": bool(transcript_match),
        "stress_match": (float(f) == 1.0),
        "failure_type": failure_type,
    }

# ----------------------------- Core Loop -----------------------------

def _load_input(input_path: Path) -> list[dict]:
    """Load items from a JSONL file or a HuggingFace Arrow dataset directory."""
    if input_path.is_dir():
        from datasets import load_from_disk
        ds = load_from_disk(str(input_path))
        return [dict(row) for row in ds]
    with jsonlines.open(input_path, "r") as r:
        return list(r)


def synthesize_and_score(
    input_path: Path,
    output_dir: Path,
    synthesize_fn: Callable,
    whistress,
    max_tries: int = 3,
    stop_when_perfect: bool = False,
    overwrite: bool = False,
):
    """
    Main synthesis + WhiStress scoring loop.

    For each item in input_path (JSONL or HF Arrow dataset dir), makes up to
    max_tries synthesis attempts, scores each with WhiStress, and writes results
    to a JSONL alongside the wavs. Supports resume: already-processed item_ids
    are skipped.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = output_dir / (input_path.stem + ".tts-whistress.jsonl")
    wav_dir = output_dir / "wavs"

    if overwrite and out_jsonl.exists():
        out_jsonl.unlink()
        print(f"Overwriting existing output: {out_jsonl}")
        if wav_dir.exists():
            shutil.rmtree(wav_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)

    processed_ids = set()
    if out_jsonl.exists():
        with jsonlines.open(out_jsonl, "r") as reader:
            for obj in reader:
                if "item_id" in obj:
                    processed_ids.add(obj["item_id"])
    print(f"Resuming: {len(processed_ids)} already processed items.")

    rows = _load_input(input_path)
    with jsonlines.open(out_jsonl, "a") as writer:
        for row in tqdm(rows, desc="Synthesizing"):
            item_id = row.get("item_id")
            if item_id and item_id in processed_ids:
                continue

            sentence = row.get("sentence", "")
            clean_sentence = sentence.replace("**", "")
            stressed_word = row.get("stressed_word", "")

            tokens = [m.group(0) for m in word_matches(clean_sentence)]
            ref_mask = [0] * len(tokens)
            try:
                tgt_idx = as_int(row.get("stressed_index"))
                if tgt_idx is not None and 0 <= tgt_idx < len(tokens):
                    ref_mask[tgt_idx] = 1
                else:
                    norm_target = normalize_word(stressed_word)
                    for i, t in enumerate(tokens):
                        if normalize_word(t) == norm_target:
                            ref_mask[i] = 1
                            break
            except Exception:
                pass

            samples = []
            safe_id = item_id if item_id else f"row_{len(processed_ids)}"

            for try_idx in range(max_tries):
                wav_name = f"{safe_id}_try{try_idx:02d}.wav"
                wav_path = wav_dir / wav_name

                if not wav_path.exists():
                    instruction = build_instruction(stressed_word, try_idx)
                    tts_input = to_single_asterisk_markups(sentence)
                    try:
                        synthesize_fn(tts_input, instruction, wav_path)
                    except Exception as e:
                        print(f"TTS Error [{safe_id} T{try_idx}]: {e}")
                        continue
                else:
                    instruction = build_instruction(stressed_word, try_idx)

                eval_res = run_whistress(whistress, wav_path, clean_sentence, ref_mask)
                samples.append({
                    "try_idx": try_idx,
                    "instruction": instruction,
                    "audio_path": str(wav_path),
                    "whistress": eval_res,
                })

                f1 = eval_res.get("metrics", {}).get("f1", 0.0)
                print(f"[{safe_id} T{try_idx}] F1: {f1:.2f} | Match: {eval_res.get('transcript_match')}")

                if stop_when_perfect and f1 == 1.0 and eval_res.get("transcript_match"):
                    break

            out_row = row.copy()
            out_row["tts_samples"] = samples
            writer.write(out_row)
            processed_ids.add(item_id)

# ----------------------------- CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Synthesize CAST items with OpenAI TTS and score with WhiStress"
    )
    ap.add_argument("input_path",
                    help="Path to validated JSONL or a HuggingFace Arrow dataset directory")
    ap.add_argument("--output-dir", default=None, help="Output directory (defaults to input's parent)")
    ap.add_argument("--voice", default="alloy", help="OpenAI TTS voice (default: alloy)")
    ap.add_argument("--model", default="gpt-4o-mini-tts", help="OpenAI TTS model (default: gpt-4o-mini-tts)")
    ap.add_argument("--max-tries", type=int, default=3, help="Synthesis attempts per item (default: 3)")
    ap.add_argument("--stop-when-perfect", action="store_true",
                    help="Stop attempts early if WhiStress F1=1.0")
    ap.add_argument("--device", default="cuda", help="Device for WhiStress (default: cuda)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    args = ap.parse_args()

    in_path = Path(args.input_path)
    output_dir = Path(args.output_dir) if args.output_dir else in_path.parent

    synthesize_fn = make_openai_synthesize_fn(model=args.model, voice=args.voice)
    whistress = WhiStressInferenceClient(device=args.device)

    synthesize_and_score(
        input_path=in_path,
        output_dir=output_dir,
        synthesize_fn=synthesize_fn,
        whistress=whistress,
        max_tries=args.max_tries,
        stop_when_perfect=args.stop_when_perfect,
        overwrite=args.overwrite,
    )

if __name__ == "__main__":
    main()
