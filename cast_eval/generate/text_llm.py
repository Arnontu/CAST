"""Text-only LLM generator for CAST evaluation."""
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from cast_eval.generate.base import TTSGenerator
from cast_eval.metrics import calc_stress_metrics, compute_group_scores, get_binary_and_words_from_item
from utils.text_utils import (
    get_formatted_prompt,
    get_response,
    parse_response,
    system_prompt_builder,
)


class TextLLMGenerator(TTSGenerator):
    """
    Text-only LLM stress prediction generator.

    Instead of generating audio, this generator runs an LLM that predicts which word
    in each sentence should be stressed given the context. Predictions are cached as
    <item_id>.json files in output_dir.

    evaluate_audio_metrics() does not apply here — use run_text_eval() instead.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        device: str = "cuda",
        n_few_shot: int = 0,
        random_fewshot: bool = False,
        thinking_model: bool = False,
        within_context: bool = False,
        mode: str = "words",
    ):
        self.model_name = model_name
        self.device = device
        self.n_few_shot = n_few_shot
        self.random_fewshot = random_fewshot
        self.thinking_model = thinking_model
        self.within_context = within_context
        self.mode = mode

    def generate(self, dataset, output_dir: Path, **kwargs) -> None:
        """Write per-sample .json prediction files; does not produce audio."""
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16
        ).to(self.device)

        system_prompt = system_prompt_builder(
            mode=self.mode, justification=False, within_context=self.within_context
        )

        for i, sample in enumerate(tqdm(dataset, desc=f"TextLLM({self.model_name})")):
            sample_id = sample["item_id"]
            pred_path = output_dir / f"{sample_id}.json"
            if pred_path.exists():
                continue

            formatted_prompt = get_formatted_prompt(
                tokenizer, dataset, sample, i,
                n_few_shot=self.n_few_shot,
                mode=self.mode,
                random_fewshot=self.random_fewshot,
                justification=False,
                system_prompt=system_prompt,
                thinking_model=self.thinking_model,
                within_context=self.within_context,
            )
            inputs = {k: v.to(self.device) for k, v in tokenizer(formatted_prompt, return_tensors="pt").items()}
            try:
                response = get_response(model, tokenizer, "pt", formatted_prompt, inputs, thinking_model=self.thinking_model)
                pred_binary, pred_word = parse_response(response, sample['sentence'], mode=self.mode)
                f1, p, r = calc_stress_metrics(sample, y_pred=pred_binary)
            except Exception as e:
                print(f"[WARN] Sample {i} ({sample_id}): {e}")
                f1, p, r = 0.0, 0.0, 0.0
                pred_word = []
                response = ""

            pred_result = {
                "item_id": sample_id,
                "gt_sentence": sample.get("sentence"),
                "context": sample.get("context"),
                "ground_truth_stressed_words": [sample.get("stressed_word")],
                "predicted_stressed_words": pred_word if isinstance(pred_word, list) else list(pred_word) if pred_word else [],
                "raw_response": response,
                "precision": p,
                "recall": r,
                "f1": f1,
            }
            pred_path.write_text(json.dumps(pred_result, indent=2), encoding='utf-8')


def run_text_eval(output_dir: Path, eval_set) -> dict:
    """
    Aggregate per-sample .json predictions in output_dir into summary metrics.

    Called after TextLLMGenerator.generate() completes.
    """
    all_results = []
    total_f1 = total_prec = total_rec = 0.0

    for item in eval_set:
        sample_id = item["item_id"]
        pred_path = output_dir / f"{sample_id}.json"
        if not pred_path.exists():
            print(f"Missing prediction: {pred_path}")
            continue
        cached = json.loads(pred_path.read_text(encoding='utf-8'))
        f1, p, r = cached["f1"], cached["precision"], cached["recall"]
        total_f1 += f1
        total_prec += p
        total_rec += r
        all_results.append({
            "audio_file": sample_id,
            "gt_sentence": cached.get("gt_sentence"),
            "ground_truth_stressed_words": cached.get("ground_truth_stressed_words"),
            "predicted_stressed_words": cached.get("predicted_stressed_words"),
            "precision": p,
            "recall": r,
            "f1": f1,
        })

    n = len(eval_set)
    Correct, Contrast, Hit, stats = compute_group_scores(
        all_results, use_alignment=False
    )
    metrics = {
        "f1": total_f1 / n if n else 0.0,
        "precision": total_prec / n if n else 0.0,
        "recall": total_rec / n if n else 0.0,
        "Correct": Correct,
        "Contrast": Contrast,
        "Hit": Hit,
        "group_stats": stats,
    }
    print("\nText LLM Evaluation Summary:")
    for k, v in metrics.items():
        if k != "group_stats":
            print(f"  {k}: {v:.4f}")
    return metrics
