"""OpenAI TTS generator for CAST evaluation."""
import json
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.transforms as T
from openai import OpenAI
from tqdm import tqdm

from cast_eval.generate.base import TTSGenerator
from utils.segmentation import prepare_text_context_mode

TARGET_SR = 16_000


class OpenAITTSGenerator(TTSGenerator):
    """
    OpenAI TTS generator for CAST evaluation.

    Unlike dataset_construction/speech/synthesize_openai.py (which does multi-try + WhiStress
    filtering to build training data), this generator does a single synthesis pass per item
    for benchmark evaluation.

    Modes: context-concat (default), explicit, tts (no context)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini-tts",
        voice: str = "alloy",
        gen_mode: str = "context-concat",
        separator: str = " ",
        max_context_length: int = 2500,
        device: str = "cuda",
        whisper_model=None,
    ):
        self.model = model
        self.voice = voice
        self.gen_mode = gen_mode
        self.separator = separator
        self.max_context_length = max_context_length
        self.device = device
        self.whisper_model = whisper_model

    def _synthesize(self, client: OpenAI, text: str, instruction: str, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with client.audio.speech.with_streaming_response.create(
                model=self.model, voice=self.voice, input=text,
                instructions=instruction, response_format="wav",
            ) as resp:
                resp.stream_to_file(str(out_path))
        except TypeError:
            with client.audio.speech.with_streaming_response.create(
                model=self.model, voice=self.voice, input=text, response_format="wav",
            ) as resp:
                resp.stream_to_file(str(out_path))
        # Resample to 16kHz
        audio, sr = sf.read(str(out_path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            t = torch.from_numpy(audio).unsqueeze(0).float()
            audio = T.Resample(orig_freq=sr, new_freq=TARGET_SR)(t).squeeze(0).numpy()
            sf.write(str(out_path), audio, TARGET_SR, subtype="PCM_16")

    def generate(self, dataset, output_dir: Path, **kwargs) -> None:
        client = OpenAI()
        output_dir.mkdir(parents=True, exist_ok=True)
        stats = {"total": len(dataset), "ok": 0, "fail": 0}

        for idx, item in enumerate(tqdm(dataset, desc="OpenAI TTS")):
            item_id = item.get('item_id', f"sample_{idx:04d}")
            base_name = Path(item.get('file_name', f"{item_id}.wav")).stem
            sentence = item.get('sentence', '').strip()
            context = item.get('context', '').strip()
            stressed_word = item.get('stressed_word', '')
            if not sentence:
                stats['fail'] += 1
                continue

            clean_sentence = sentence.replace('**', '').replace('*', '')

            if self.gen_mode == "tts":
                tts_text = clean_sentence
                instruction = "Read naturally."
            elif self.gen_mode == "explicit":
                tts_text = clean_sentence
                instruction = f"Read naturally. Emphasize '{stressed_word}' clearly." if stressed_word else "Read naturally."
            else:  # context-concat
                tts_text, actual_context, _ = prepare_text_context_mode(
                    sentence, context, self.separator, self.max_context_length
                )
                instruction = "Read naturally, reflecting the context's implied stress."

            out_path = output_dir / f"{base_name}_00.wav"
            try:
                self._synthesize(client, tts_text, instruction, out_path)
            except Exception as e:
                print(f"OpenAI TTS error [{base_name}]: {e}")
                stats['fail'] += 1
                continue

            if self.whisper_model:
                try:
                    audio, _ = sf.read(str(out_path))
                    out = self.whisper_model(audio, generate_kwargs={"language": "en"})
                    (output_dir / f"{base_name}_00.trans").write_text(out['text'].strip(), encoding='utf-8')
                except Exception as e:
                    print(f"Whisper error [{base_name}]: {e}")

            (output_dir / f"{base_name}_00.txt").write_text(tts_text, encoding='utf-8')
            stats['ok'] += 1

        json.dump(stats, (output_dir / "generation_stats.json").open('w'), indent=2)
        print(f"OpenAI TTS done: {stats['ok']}/{stats['total']} ok, {stats['fail']} fail")
