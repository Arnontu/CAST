"""CosyVoice TTS generator for CAST evaluation."""
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


class CosyVoiceGenerator(TTSGenerator):
    """
    CosyVoice TTS generator.

    Modes:
        context-concat: context prepended to sentence (with Whisper segmentation)
        context-instruction: context passed as instruction parameter
        tts: plain TTS, no context
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "CosyVoice-300M",
        speed: float = 1.0,
        separator: str = " ",
        max_context_length: int = 2500,
        instruction_template: str = None,
        concat_instruction: str = None,
        whisper_model=None,
        gen_mode: str = "context-concat",
    ):
        self.device = device
        self.model_name = model_name
        self.speed = speed
        self.separator = separator
        self.max_context_length = max_context_length
        self.instruction_template = instruction_template
        self.concat_instruction = concat_instruction
        self.whisper_model = whisper_model
        self.gen_mode = gen_mode

    def generate(self, dataset, output_dir: Path, **kwargs) -> None:
        from cosyvoice.cli.cosyvoice import CosyVoice
        model = CosyVoice(self.model_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        stats = {"total": len(dataset), "ok": 0, "fail": 0, "seg_fail": 0}

        for idx, item in enumerate(tqdm(dataset, desc="CosyVoice")):
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
                instruction = None
                needs_segmentation = False
                actual_context = ""
            elif self.gen_mode == "context-instruction":
                tts_text = clean_sentence
                if self.instruction_template and context:
                    instruction = self.instruction_template.format(context=context)
                else:
                    instruction = context or None
                needs_segmentation = False
                actual_context = context
            else:  # context-concat
                full_text, actual_context, needs_segmentation = prepare_text_context_mode(
                    sentence, context, self.separator, self.max_context_length
                )
                tts_text = full_text
                instruction = self.concat_instruction

            try:
                if instruction:
                    output = model.inference_instruct(tts_text, instruction=instruction, speed=self.speed)
                else:
                    output = model.inference_sft(tts_text, speed=self.speed)
                audio_chunks = list(output)
                if not audio_chunks:
                    raise ValueError("No audio generated")
                wav = np.concatenate([c['tts_speech'].numpy().squeeze() for c in audio_chunks])
                model_sr = 22050
            except Exception as e:
                print(f"CosyVoice error [{base_name}]: {e}")
                stats['fail'] += 1
                continue

            segmented_wav = wav
            full_trans = segmented_trans = ""

            if needs_segmentation and self.whisper_model:
                try:
                    import torch
                    wav_16k = F.resample(torch.from_numpy(wav).float(), orig_freq=model_sr, new_freq=16000).numpy()
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
                    import torch
                    wav_16k = F.resample(torch.from_numpy(wav).float(), orig_freq=model_sr, new_freq=16000).numpy()
                    out = self.whisper_model(wav_16k, generate_kwargs={"language": "en"})
                    full_trans = segmented_trans = out['text'].strip()
                except Exception as e:
                    print(f"Whisper error [{base_name}]: {e}")

            if model_sr != TARGET_SR:
                import torch
                segmented_wav = F.resample(torch.from_numpy(segmented_wav).float(), orig_freq=model_sr, new_freq=TARGET_SR).numpy()

            sf.write(str(output_dir / f"{base_name}_00.wav"), segmented_wav, TARGET_SR)
            (output_dir / f"{base_name}_00.trans").write_text(segmented_trans, encoding='utf-8')
            (output_dir / f"{base_name}_00.whisper").write_text(full_trans, encoding='utf-8')
            (output_dir / f"{base_name}_00.txt").write_text(tts_text, encoding='utf-8')
            stats['ok'] += 1

        json.dump(stats, (output_dir / "generation_stats.json").open('w'), indent=2)
        print(f"CosyVoice done: {stats['ok']}/{stats['total']} ok, {stats['fail']} fail, {stats['seg_fail']} seg_fail")
