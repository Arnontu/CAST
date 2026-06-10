"""Kokoro TTS generator for CAST evaluation."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torchaudio.functional as F
from tqdm import tqdm

from cast_eval.generate.base import TTSGenerator
from utils.segmentation import (
    normalizer,
    find_subsequence_fuzzy,
    prepare_text_context_mode,
    segment_text_portion,
)

TARGET_SR = 16_000


class KokoroGenerator(TTSGenerator):
    """
    Kokoro TTS generator.

    Modes: context-concat (default), tts (no context)
    """

    def __init__(
        self,
        device: str = "cuda",
        voice: str = "af_heart",
        speed: float = 1.0,
        lang_code: str = "a",
        separator: str = " ",
        max_context_length: int = 2500,
        whisper_model=None,
        gen_mode: str = "context-concat",
    ):
        self.device = device
        self.voice = voice
        self.speed = speed
        self.lang_code = lang_code
        self.separator = separator
        self.max_context_length = max_context_length
        self.whisper_model = whisper_model
        self.gen_mode = gen_mode

    def generate(self, dataset, output_dir: Path, **kwargs) -> None:
        from kokoro import KPipeline
        pipeline = KPipeline(lang_code=self.lang_code, device=self.device)
        output_dir.mkdir(parents=True, exist_ok=True)
        stats = {"total": len(dataset), "ok": 0, "fail": 0, "seg_fail": 0}
        no_context = self.gen_mode == "tts"

        for idx, item in enumerate(tqdm(dataset, desc="Kokoro")):
            item_id = item.get('item_id', f"sample_{idx:04d}")
            base_name = Path(item.get('file_name', f"{item_id}.wav")).stem
            sentence = item.get('sentence', '').strip()
            context = item.get('context', '').strip()
            if not sentence:
                stats['fail'] += 1
                continue

            clean_sentence = sentence.replace('**', '').replace('*', '')

            if no_context:
                full_text = clean_sentence
                needs_segmentation = False
                actual_context = ""
            else:
                full_text, actual_context, needs_segmentation = prepare_text_context_mode(
                    sentence, context, self.separator, self.max_context_length
                )

            try:
                generator = pipeline(full_text, voice=self.voice, speed=self.speed, split_pattern=None)
                audio_chunks = [chunk.audio for chunk in generator if chunk.audio is not None]
                if not audio_chunks:
                    raise ValueError("No audio generated")
                wav = np.concatenate(audio_chunks)
                model_sr = 24000
            except Exception as e:
                print(f"Kokoro error [{base_name}]: {e}")
                stats['fail'] += 1
                continue

            segmented_wav = wav
            full_trans = segmented_trans = ""

            if needs_segmentation and self.whisper_model:
                try:
                    wav_16k = F.resample(
                        __import__('torch').from_numpy(wav).float(), orig_freq=model_sr, new_freq=16000
                    ).numpy()
                    out = self.whisper_model(wav_16k, return_timestamps="word", generate_kwargs={"language": "en"})
                    full_trans = out['text'].strip()
                    chunks = out.get('chunks', [])
                    seg = segment_text_portion(wav, model_sr, clean_sentence, full_trans, chunks, actual_context, self.separator)
                    if seg is None:
                        segmented_wav = wav
                        segmented_trans = full_trans
                        stats['seg_fail'] += 1
                    else:
                        segmented_wav = seg
                        target_words = normalizer(clean_sentence).split()
                        full_words = normalizer(full_trans).split()
                        mi = find_subsequence_fuzzy(full_words, target_words, threshold=0.7)
                        segmented_trans = " ".join(full_words[mi: mi + len(target_words)]) if mi is not None else full_trans
                except Exception as e:
                    print(f"Segmentation error [{base_name}]: {e}")
                    segmented_wav = wav
                    segmented_trans = full_trans
                    stats['seg_fail'] += 1
            elif self.whisper_model:
                try:
                    wav_16k = F.resample(
                        __import__('torch').from_numpy(wav).float(), orig_freq=model_sr, new_freq=16000
                    ).numpy()
                    out = self.whisper_model(wav_16k, generate_kwargs={"language": "en"})
                    full_trans = segmented_trans = out['text'].strip()
                except Exception as e:
                    print(f"Whisper error [{base_name}]: {e}")

            if model_sr != TARGET_SR:
                segmented_wav = F.resample(
                    __import__('torch').from_numpy(segmented_wav).float(), orig_freq=model_sr, new_freq=TARGET_SR
                ).numpy()

            sf.write(str(output_dir / f"{base_name}_00.wav"), segmented_wav, TARGET_SR)
            (output_dir / f"{base_name}_00.trans").write_text(segmented_trans, encoding='utf-8')
            (output_dir / f"{base_name}_00.whisper").write_text(full_trans, encoding='utf-8')
            (output_dir / f"{base_name}_00.txt").write_text(full_text, encoding='utf-8')
            stats['ok'] += 1

        json.dump(stats, (output_dir / "generation_stats.json").open('w'), indent=2)
        print(f"Kokoro done: {stats['ok']}/{stats['total']} ok, {stats['fail']} fail, {stats['seg_fail']} seg_fail")
