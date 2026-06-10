"""Qwen3-TTS generator for CAST evaluation."""
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

QWEN3TTS_SPEAKERS = [
    "Chelsie", "Ethan", "Alex", "River", "Aria", "Ryan",
    "Valentina", "Marcus", "Zoe", "Liam",
]


def _resolve_model_id(model_name: str) -> str:
    if "/" in model_name:
        return model_name
    return f"Qwen/Qwen3-TTS-{model_name}"


class Qwen3TTSGenerator(TTSGenerator):
    """
    Qwen3-TTS CustomVoice generator.

    Modes: tts, context-concat, context-instruction, explicit
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "1.7B",
        speaker: str = "Ryan",
        separator: str = " ",
        max_context_length: int = 2500,
        instruction_template: str = None,
        whisper_model=None,
        gen_mode: str = "context-concat",
    ):
        self.device = device
        self.model_id = _resolve_model_id(model_name)
        self.speaker = speaker
        self.separator = separator
        self.max_context_length = max_context_length
        self.instruction_template = instruction_template
        self.whisper_model = whisper_model
        self.gen_mode = gen_mode

    def generate(self, dataset, output_dir: Path, **kwargs) -> None:
        import torch
        from transformers import AutoTokenizer, Qwen2_5OmniForConditionalGeneration
        # Qwen3-TTS CustomVoice API
        from qwen3_tts import CustomVoiceTTS  # noqa: F401 — installed with qwen3_tts package

        model = CustomVoiceTTS.from_pretrained(self.model_id, device=self.device)
        output_dir.mkdir(parents=True, exist_ok=True)
        stats = {"total": len(dataset), "ok": 0, "fail": 0, "seg_fail": 0}
        model_sr = 24000

        for idx, item in enumerate(tqdm(dataset, desc="Qwen3-TTS")):
            item_id = item.get('item_id', f"sample_{idx:04d}")
            base_name = Path(item.get('file_name', f"{item_id}.wav")).stem
            sentence = item.get('sentence', '').strip()
            context = item.get('context', '').strip()
            if not sentence:
                stats['fail'] += 1
                continue

            clean_sentence = sentence.replace('**', '').replace('*', '')

            if self.gen_mode == "tts":
                tts_text = clean_sentence
                instruct = None
                needs_segmentation = False
                actual_context = ""
            elif self.gen_mode == "explicit":
                tts_text = clean_sentence
                sw = item.get('stressed_word', '')
                instruct = f"Stress the word '{sw}'." if sw else None
                needs_segmentation = False
                actual_context = ""
            elif self.gen_mode == "context-instruction":
                tts_text = clean_sentence
                if self.instruction_template and context:
                    instruct = self.instruction_template.format(context=context)
                else:
                    instruct = context or None
                needs_segmentation = False
                actual_context = context
            else:  # context-concat
                full_text, actual_context, needs_segmentation = prepare_text_context_mode(
                    sentence, context, self.separator, self.max_context_length
                )
                tts_text = full_text
                instruct = None

            try:
                result = model.synthesize(tts_text, speaker=self.speaker, instruct=instruct)
                wav = result['audio']
                if hasattr(wav, 'numpy'):
                    wav = wav.numpy()
                wav = np.squeeze(wav)
            except Exception as e:
                print(f"Qwen3TTS error [{base_name}]: {e}")
                stats['fail'] += 1
                continue

            segmented_wav = wav
            full_trans = segmented_trans = ""

            if needs_segmentation and self.whisper_model:
                try:
                    wav_16k = F.resample(
                        torch.from_numpy(wav).float(), orig_freq=model_sr, new_freq=16000
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
                        torch.from_numpy(wav).float(), orig_freq=model_sr, new_freq=16000
                    ).numpy()
                    out = self.whisper_model(wav_16k, generate_kwargs={"language": "en"})
                    full_trans = segmented_trans = out['text'].strip()
                except Exception as e:
                    print(f"Whisper error [{base_name}]: {e}")

            if model_sr != TARGET_SR:
                segmented_wav = F.resample(
                    torch.from_numpy(segmented_wav).float(), orig_freq=model_sr, new_freq=TARGET_SR
                ).numpy()

            sf.write(str(output_dir / f"{base_name}_00.wav"), segmented_wav, TARGET_SR)
            (output_dir / f"{base_name}_00.trans").write_text(segmented_trans, encoding='utf-8')
            (output_dir / f"{base_name}_00.whisper").write_text(full_trans, encoding='utf-8')
            (output_dir / f"{base_name}_00.txt").write_text(tts_text, encoding='utf-8')
            stats['ok'] += 1

        json.dump(stats, (output_dir / "generation_stats.json").open('w'), indent=2)
        print(f"Qwen3TTS done: {stats['ok']}/{stats['total']} ok, {stats['fail']} fail, {stats['seg_fail']} seg_fail")
