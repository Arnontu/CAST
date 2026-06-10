"""HiggsAudio V2 generator for CAST evaluation."""
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

HIGGSAUDIO_MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
HIGGSAUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

MULTISPEAKER_SYSTEM = (
    "You are an AI assistant designed to convert text into speech.\n"
    "If the user's message includes a [SPEAKER*] tag, do not read out the tag "
    "and generate speech for the following text, using the specified voice.\n"
    "If no speaker tag is present, select a suitable voice on your own."
)

TARGET_SR = 16_000


class HiggsAudioGenerator(TTSGenerator):
    """
    HiggsAudio V2 generator.

    Modes: tts, context-concat, context-instruction, explicit, dialog
    """

    def __init__(
        self,
        device: str = "cuda",
        model_path: str = HIGGSAUDIO_MODEL_PATH,
        tokenizer_path: str = HIGGSAUDIO_TOKENIZER_PATH,
        gen_mode: str = "tts",
        separator: str = " ",
        max_context_length: int = 2500,
        instruction_template: str = None,
        temperature: float = 0.3,
        top_p: float = 0.95,
        top_k: int = 50,
        max_new_tokens: int = 2048,
        dialog_spk0_desc: str = "feminine",
        dialog_spk1_desc: str = "masculine",
        dialog_stress_instruction: str = "Apply vocal emphasis to exactly one word based on the conversational context from SPEAKER0.",
        ref_audio_path: str = None,
        ref_audio_text: str = None,
        whisper_model=None,
    ):
        self.device = device
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.gen_mode = gen_mode
        self.separator = separator
        self.max_context_length = max_context_length
        self.instruction_template = instruction_template
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.dialog_spk0_desc = dialog_spk0_desc
        self.dialog_spk1_desc = dialog_spk1_desc
        self.dialog_stress_instruction = dialog_stress_instruction
        self.ref_audio_path = ref_audio_path
        self.ref_audio_text = ref_audio_text
        self.whisper_model = whisper_model

    def generate(self, dataset, output_dir: Path, **kwargs) -> None:
        try:
            from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
            from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
        except ImportError:
            raise ImportError("boson_multimodal not found. See https://github.com/boson-ai/higgs-audio")

        engine = HiggsAudioServeEngine(self.model_path, self.tokenizer_path, device=self.device)
        output_dir.mkdir(parents=True, exist_ok=True)
        stats = {"total": len(dataset), "ok": 0, "fail": 0, "seg_fail": 0}
        gen_mode = self.gen_mode

        for idx, item in enumerate(tqdm(dataset, desc="HiggsAudio")):
            item_id = item.get('item_id', f"sample_{idx:04d}")
            base_name = Path(item.get('file_name', f"{item_id}.wav")).stem
            sentence = item.get('sentence', '').strip()
            context = item.get('context', '').strip()
            if not sentence:
                stats['fail'] += 1
                continue

            clean_sentence = sentence.replace('**', '').replace('*', '')

            if gen_mode == "tts":
                tts_text = clean_sentence
                system_content = "Generate audio following instruction."
                needs_segmentation = False
                actual_context = ""
            elif gen_mode == "explicit":
                tts_text = clean_sentence
                sw = item.get('stressed_word', '')
                system_content = f"Generate audio following instruction. Stress the word '{sw}'." if sw else "Generate audio following instruction."
                needs_segmentation = False
                actual_context = ""
            elif gen_mode == "context-instruction":
                tts_text = clean_sentence
                if self.instruction_template and context:
                    instruction = self.instruction_template.format(context=context)
                else:
                    instruction = context or ""
                system_content = f"Generate audio following instruction. {instruction}".strip()
                needs_segmentation = False
                actual_context = context
            elif gen_mode == "context-concat":
                tts_text, actual_context, needs_segmentation = prepare_text_context_mode(
                    sentence, context, self.separator, self.max_context_length
                )
                system_content = "Generate audio following instruction."
            elif gen_mode == "dialog":
                actual_context = context[:self.max_context_length] if context else ""
                if not actual_context:
                    tts_text = clean_sentence
                    system_content = "Generate audio following instruction."
                    needs_segmentation = False
                else:
                    spk1_desc = self.dialog_spk1_desc
                    if self.dialog_stress_instruction:
                        spk1_desc += f", {self.dialog_stress_instruction}"
                    system_content = (
                        f"{MULTISPEAKER_SYSTEM}\n\n<|scene_desc_start|>\n"
                        f"SPEAKER0: {self.dialog_spk0_desc}\nSPEAKER1: {spk1_desc}\n<|scene_desc_end|>"
                    )
                    tts_text = f"[SPEAKER0]\n{actual_context}\n\n[SPEAKER1]\n{clean_sentence}"
                    needs_segmentation = True
            else:
                print(f"Unknown gen_mode: {gen_mode}")
                stats['fail'] += 1
                continue

            messages = [Message(role="system", content=system_content)]
            if self.ref_audio_path and self.ref_audio_text and gen_mode != "dialog":
                messages.append(Message(role="user", content=self.ref_audio_text))
                messages.append(Message(role="assistant", content=AudioContent(audio_url=self.ref_audio_path)))
            messages.append(Message(role="user", content=tts_text))

            try:
                out = engine.generate(
                    chat_ml_sample=ChatMLSample(messages=messages),
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                )
                import torch
                wav = out.audio
                if isinstance(wav, torch.Tensor):
                    wav = wav.cpu().numpy()
                wav = np.squeeze(wav)
                model_sr = out.sampling_rate
            except Exception as e:
                print(f"HiggsAudio error [{base_name}]: {e}")
                stats['fail'] += 1
                continue

            segmented_wav = wav
            full_trans = segmented_trans = None

            if needs_segmentation and self.whisper_model:
                try:
                    import torch
                    wav_16k = F.resample(torch.from_numpy(wav).float(), orig_freq=model_sr, new_freq=16000).numpy() if model_sr != 16000 else wav
                    wo = self.whisper_model(wav_16k, return_timestamps="word", generate_kwargs={"language": "en"})
                    full_trans = wo['text'].strip()
                    chunks = wo.get('chunks', [])
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
                    wav_16k = F.resample(torch.from_numpy(wav).float(), orig_freq=model_sr, new_freq=16000).numpy() if model_sr != 16000 else wav
                    wo = self.whisper_model(wav_16k, generate_kwargs={"language": "en"})
                    full_trans = segmented_trans = wo['text'].strip()
                except Exception as e:
                    print(f"Whisper error [{base_name}]: {e}")

            if model_sr != TARGET_SR:
                import torch
                segmented_wav = F.resample(torch.from_numpy(segmented_wav).float(), orig_freq=model_sr, new_freq=TARGET_SR).numpy()

            sf.write(str(output_dir / f"{base_name}_00.wav"), segmented_wav, TARGET_SR)
            if segmented_trans is not None:
                (output_dir / f"{base_name}_00.trans").write_text(segmented_trans, encoding='utf-8')
            if full_trans is not None:
                (output_dir / f"{base_name}_00.whisper").write_text(full_trans, encoding='utf-8')
            (output_dir / f"{base_name}_00.txt").write_text(tts_text, encoding='utf-8')
            stats['ok'] += 1

        json.dump(stats, (output_dir / "generation_stats.json").open('w'), indent=2)
        print(f"HiggsAudio done: {stats['ok']}/{stats['total']} ok, {stats['fail']} fail, {stats['seg_fail']} seg_fail")
