"""Chatterbox TTS generator for CAST evaluation."""
import json
from pathlib import Path

import soundfile as sf
import torchaudio.functional as F
from tqdm import tqdm

from cast_eval.generate.base import TTSGenerator
from utils.segmentation import (
    normalizer,
    find_subsequence_fuzzy,
    prepare_text_context_mode,
    prepare_text_explicit_mode,
    segment_text_portion,
)


def _load_chatterbox(model_variant: str = "standard", device: str = "cuda"):
    if model_variant == "turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        return ChatterboxTurboTTS.from_pretrained(device=device)
    elif model_variant == "multilingual":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        return ChatterboxMultilingualTTS.from_pretrained(device=device)
    else:
        from chatterbox.tts import ChatterboxTTS
        return ChatterboxTTS.from_pretrained(device=device)


class ChatterboxGenerator(TTSGenerator):
    """
    Chatterbox TTS generator.

    Modes:
        context-concat: prepend context to sentence; Whisper segments the sentence portion
        explicit: insert pause tokens around the stressed word
        tts: plain TTS on the sentence, no context
    """

    def __init__(
        self,
        device: str = "cuda",
        model_variant: str = "standard",
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        separator: str = " ",
        max_context_length: int = 2500,
        pause_token: str = "[pause:0.25s]",
        whisper_model=None,
        gen_mode: str = "context-concat",
    ):
        self.device = device
        self.model_variant = model_variant
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.temperature = temperature
        self.separator = separator
        self.max_context_length = max_context_length
        self.pause_token = pause_token
        self.whisper_model = whisper_model
        self.gen_mode = gen_mode

    def generate(self, dataset, output_dir: Path, **kwargs) -> None:
        explicit_mode = self.gen_mode == "explicit"
        no_context = self.gen_mode == "tts"
        mv = self.model_variant
        if explicit_mode and mv == "standard":
            mv = "turbo"
        model = _load_chatterbox(mv, self.device)
        target_sr = 16_000
        output_dir.mkdir(parents=True, exist_ok=True)
        stats = {"total": len(dataset), "ok": 0, "fail": 0, "seg_fail": 0}

        for idx, item in enumerate(tqdm(dataset, desc="Chatterbox")):
            item_id = item.get('item_id', f"sample_{idx:04d}")
            base_name = Path(item.get('file_name', f"{item_id}.wav")).stem
            sentence = item.get('sentence', '').strip()
            context = item.get('context', '').strip()
            stressed_index = item.get('stressed_index', -1)
            if not sentence:
                stats['fail'] += 1
                continue

            if explicit_mode:
                full_text = prepare_text_explicit_mode(sentence, stressed_index, self.pause_token)
                needs_segmentation = False
                actual_context = ""
            elif no_context:
                full_text = sentence.replace('**', '').replace('*', '')
                needs_segmentation = False
                actual_context = ""
            else:
                full_text, actual_context, needs_segmentation = prepare_text_context_mode(
                    sentence, context, self.separator, self.max_context_length
                )

            try:
                wav = model.generate(
                    text=full_text,
                    exaggeration=self.exaggeration,
                    cfg_weight=self.cfg_weight,
                    temperature=self.temperature,
                )
                if wav.ndim > 1:
                    wav = wav.squeeze()
            except Exception as e:
                print(f"Chatterbox error [{base_name}]: {e}")
                stats['fail'] += 1
                continue

            clean_sentence = sentence.replace('**', '').replace('*', '')
            segmented_wav = wav
            full_trans = segmented_trans = ""

            if needs_segmentation and self.whisper_model:
                try:
                    wav_16k = F.resample(wav.float(), orig_freq=model.sr, new_freq=16000).numpy() if model.sr != 16000 else wav
                    out = self.whisper_model(wav_16k, return_timestamps="word", generate_kwargs={"language": "en"})
                    full_trans = out['text'].strip()
                    chunks = out.get('chunks', [])
                    seg = segment_text_portion(wav, model.sr, clean_sentence, full_trans, chunks, actual_context, self.separator)
                    if seg is None:
                        segmented_wav = wav
                        segmented_trans = full_trans
                        stats['seg_fail'] += 1
                    else:
                        segmented_wav = seg
                        target_words = normalizer(clean_sentence).split()
                        full_words = normalizer(full_trans).split()
                        mi = find_subsequence_fuzzy(full_words, target_words, threshold=0.7)
                        if mi is not None:
                            segmented_trans = " ".join(full_words[mi: mi + len(target_words)])
                        else:
                            segmented_trans = full_trans
                except Exception as e:
                    print(f"Segmentation error [{base_name}]: {e}")
                    segmented_wav = wav
                    segmented_trans = full_trans
                    stats['seg_fail'] += 1
            elif self.whisper_model:
                try:
                    wav_16k = F.resample(wav.float(), orig_freq=model.sr, new_freq=16000).numpy() if model.sr != 16000 else wav
                    out = self.whisper_model(wav_16k, generate_kwargs={"language": "en"})
                    full_trans = segmented_trans = out['text'].strip()
                except Exception as e:
                    print(f"Whisper error [{base_name}]: {e}")

            if model.sr != target_sr:
                segmented_wav = F.resample(segmented_wav.float(), orig_freq=model.sr, new_freq=target_sr).numpy()

            out_path = output_dir / f"{base_name}_00.wav"
            sf.write(str(out_path), segmented_wav, target_sr)
            (output_dir / f"{base_name}_00.trans").write_text(segmented_trans, encoding='utf-8')
            (output_dir / f"{base_name}_00.whisper").write_text(full_trans, encoding='utf-8')
            (output_dir / f"{base_name}_00.txt").write_text(full_text, encoding='utf-8')
            stats['ok'] += 1

        json.dump(stats, (output_dir / "generation_stats.json").open('w'), indent=2)
        print(f"Chatterbox done: {stats['ok']}/{stats['total']} ok, {stats['fail']} fail, {stats['seg_fail']} seg_fail")
