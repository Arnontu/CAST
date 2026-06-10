#!/usr/bin/env python3
"""Unified evaluation CLI for the CAST benchmark."""
import argparse
import json
from pathlib import Path

from transformers import pipeline as hf_pipeline

from cast_eval.dataset import load_test_dataset
from cast_eval.metrics import evaluate_audio_metrics


# ---------------------------------------------------------------------------
# Generator factory
# ---------------------------------------------------------------------------

def build_generator(args, whisper_model=None):
    model_key = args.model.lower()

    if model_key == "chatterbox":
        from cast_eval.generate.chatterbox import ChatterboxGenerator
        return ChatterboxGenerator(
            device=args.device,
            model_variant=args.model_variant or "standard",
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            temperature=args.temperature,
            separator=args.separator,
            pause_token=args.pause_token,
            whisper_model=whisper_model,
            gen_mode=args.mode,
        )

    elif model_key == "kokoro":
        from cast_eval.generate.kokoro import KokoroGenerator
        return KokoroGenerator(
            device=args.device,
            voice=args.voice or "af_heart",
            speed=args.speed,
            lang_code=args.lang_code or "a",
            separator=args.separator,
            whisper_model=whisper_model,
            gen_mode=args.mode,
        )

    elif model_key == "cosyvoice":
        from cast_eval.generate.cosyvoice import CosyVoiceGenerator
        return CosyVoiceGenerator(
            device=args.device,
            model_name=args.model_name or "CosyVoice-300M",
            speed=args.speed,
            separator=args.separator,
            instruction_template=args.instruction_template,
            whisper_model=whisper_model,
            gen_mode=args.mode,
        )

    elif model_key == "higgsaudio":
        from cast_eval.generate.higgsaudio import HiggsAudioGenerator, HIGGSAUDIO_MODEL_PATH, HIGGSAUDIO_TOKENIZER_PATH
        return HiggsAudioGenerator(
            device=args.device,
            model_path=args.model_name or HIGGSAUDIO_MODEL_PATH,
            tokenizer_path=getattr(args, 'tokenizer_path', None) or HIGGSAUDIO_TOKENIZER_PATH,
            gen_mode=args.mode,
            separator=args.separator,
            instruction_template=args.instruction_template,
            temperature=args.temperature,
            whisper_model=whisper_model,
        )

    elif model_key == "qwen3tts":
        from cast_eval.generate.qwen3tts import Qwen3TTSGenerator
        return Qwen3TTSGenerator(
            device=args.device,
            model_name=args.model_name or "1.7B",
            speaker=args.voice or "Ryan",
            separator=args.separator,
            instruction_template=args.instruction_template,
            whisper_model=whisper_model,
            gen_mode=args.mode,
        )

    elif model_key == "openai-tts":
        from cast_eval.generate.openai_tts import OpenAITTSGenerator
        return OpenAITTSGenerator(
            model=args.openai_model or "gpt-4o-mini-tts",
            voice=args.openai_voice or "alloy",
            gen_mode=args.mode,
            separator=args.separator,
            device=args.device,
            whisper_model=whisper_model,
        )

    elif model_key == "text-llm":
        from cast_eval.generate.text_llm import TextLLMGenerator
        return TextLLMGenerator(
            model_name=args.model_name or "Qwen/Qwen3-0.6B",
            device=args.device,
            n_few_shot=args.n_few_shot,
            random_fewshot=args.random_fewshot,
            thinking_model=args.thinking_model,
            within_context=args.within_context,
            mode="words",
        )

    else:
        raise ValueError(
            f"Unknown model: {args.model!r}. "
            "Choose from: chatterbox, kokoro, cosyvoice, higgsaudio, qwen3tts, openai-tts, text-llm"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="CAST benchmark evaluation")

    # Required
    ap.add_argument("--model", required=True, help="Generator model key (chatterbox, kokoro, ..., text-llm)")
    ap.add_argument("--dataset", required=True, help="Path to CAST dataset JSONL or Arrow dir")
    ap.add_argument("--output-dir", required=True, type=Path, help="Output directory for results")

    # Common
    ap.add_argument("--device", default="cuda", help="Device (default: cuda)")
    ap.add_argument("--mode", default="context-concat",
                    choices=["context-concat", "context-instruction", "explicit", "tts", "dialog"],
                    help="Generation mode (default: context-concat)")
    ap.add_argument("--separator", default=" ", help="Context-sentence separator (default: space)")
    ap.add_argument("--tasks", nargs="+", default=["wer", "stress"], choices=["wer", "stress"],
                    help="Metrics to compute (default: wer stress)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing generated audio")
    ap.add_argument("--model-name", default=None, help="HF model ID or model variant (model-specific)")
    ap.add_argument("--instruction-template", default=None,
                    help="Instruction template with {context} placeholder")

    # Chatterbox
    ap.add_argument("--exaggeration", type=float, default=0.5)
    ap.add_argument("--cfg-weight", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--model-variant", default=None, choices=["standard", "turbo", "multilingual"])
    ap.add_argument("--pause-token", default="[pause:0.25s]")

    # Kokoro / Qwen3TTS / OpenAI
    ap.add_argument("--voice", default=None, help="Voice name (kokoro: af_heart, qwen3tts: Ryan)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--lang-code", default="a")

    # OpenAI TTS
    ap.add_argument("--openai-model", default="gpt-4o-mini-tts")
    ap.add_argument("--openai-voice", default="alloy")

    # HiggsAudio
    ap.add_argument("--tokenizer-path", default=None)

    # Text LLM
    ap.add_argument("--n-few-shot", type=int, default=0)
    ap.add_argument("--random-fewshot", action="store_true")
    ap.add_argument("--thinking-model", action="store_true")
    ap.add_argument("--within-context", action="store_true")

    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    ds = load_test_dataset(args.dataset)
    print(f"Dataset size: {len(ds)}")

    is_text_llm = args.model.lower() == "text-llm"

    # Load Whisper if needed for audio models
    whisper_model = None
    if not is_text_llm and "wer" in args.tasks:
        print("Loading Whisper...")
        whisper_model = hf_pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
            device=args.device,
        )

    # Load WhiStress if needed
    whistress = None
    if not is_text_llm and "stress" in args.tasks:
        from whistress import WhiStressInferenceClient
        print("Loading WhiStress...")
        whistress = WhiStressInferenceClient(device=args.device)

    # Build generator and run
    generator = build_generator(args, whisper_model=whisper_model)
    audio_dir = args.output_dir / "audio"

    if is_text_llm:
        preds_dir = args.output_dir / "preds"
        if not preds_dir.exists() or args.overwrite:
            generator.generate(ds, preds_dir)
        from cast_eval.generate.text_llm import run_text_eval
        metrics = run_text_eval(preds_dir, ds)
    else:
        if not audio_dir.exists() or args.overwrite:
            generator.generate(ds, audio_dir)
        metrics = evaluate_audio_metrics(
            ds,
            audio_dir,
            whistress=whistress,
            whisper=whisper_model,
            calc_wer="wer" in args.tasks,
            calc_stress="stress" in args.tasks,
        )

    # Save results
    results_path = args.output_dir / "metrics.json"
    results_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
