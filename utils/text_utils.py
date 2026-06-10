"""Text prompt utilities for stress evaluation."""
import ast
import re
import logging
import random
from functools import lru_cache

import torch
from whisper_normalizer.english import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def system_prompt_builder(mode: str, justification: bool = False, within_context: bool = False) -> str:
    if within_context:
        return (
            "You are a context-aware word-stress classifier. "
            "Given a Text, identify EXACTLY ONE word in the Text that should be emphasized "
            "to reflect the intended meaning conveyed by the Context of the text."
            "Return a JSON object where `emphasized_word` is a string taken verbatim from the Text "
            "(no lemmatization, no tense/number changes, no synonyms). "
            "Respond ONLY with a JSON object in this exact format:\n"
            "{\n"
            '  "emphasized_word": ["word"]\n'
            "}\n"
            "Do not include any other text or keys."
        )

    return (
        "You are a context-aware word-stress classifier.\n\n"
        "Input:\n"
        "- Context: a preceding conversational turn\n"
        "- Text: the next utterance\n\n"
        "Task:\n"
        "Pick EXACTLY ONE word from Text that should be emphasized to reflect the meaning implied by Context.\n\n"
        "Hard constraints (must follow):\n"
        "- The output word MUST appear in Text.\n"
        "- Treat \"word in Text\" as whitespace-split tokens after stripping leading/trailing punctuation.\n"
        "- Match case-insensitively, but output the surface form exactly as it appears in Text "
        "(choose the best occurrence).\n"
        "- Do NOT output any word that appears only in Context.\n"
        "- If no valid in-Text word can be chosen, output an empty string.\n\n"
        "Return ONLY this JSON object (no extra keys, no extra text):\n"
        "{\n"
        '"emphasized_word": "<one word from Text or empty string>"\n'
        "}\n"
    )


def get_few_shot_examples(eval_dataset, current_idx: int, n: int = 3, mode: str = "words",
                           random_fewshot: bool = False, justification: bool = False) -> list:
    if n == 0:
        return []
    indices = [j for j in range(len(eval_dataset)) if j != current_idx]
    chosen = random.sample(indices, n) if random_fewshot else (indices[:n] if len(indices) >= n else indices)

    from cast_eval.metrics import get_binary_and_words_from_item

    few_shots = []
    for j in chosen:
        ex = eval_dataset[j]
        text = ex['sentence'].replace("*", "")
        emphasized_binary, emphasized_words = get_binary_and_words_from_item(ex)

        if mode == "words":
            emphasized_output = emphasized_words
        elif mode == "indices":
            emphasized_output = [i for i, s in enumerate(emphasized_binary) if s]
        elif mode == "markup":
            words = text.split()
            emphasized_output = " ".join(
                f"**{w}**" if emphasized_binary[i] else w for i, w in enumerate(words)
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        few_shots.append({
            "context": ex['context'],
            "text": text,
            "emphasized_output": emphasized_output,
            "justification": (
                "These words are emphasized to highlight the key information conveyed by the context."
                if justification else None
            ),
            "mode": mode,
        })
    return few_shots


def get_formatted_prompt(text_tokenizer, prompts_dataset, sample, i: int, n_few_shot: int,
                          mode: str, random_fewshot: bool, system_prompt: str,
                          thinking_model: bool, justification: bool = False,
                          within_context: bool = False) -> str:
    transcription = sample['sentence'].replace("*", "").strip()
    few_shots = get_few_shot_examples(prompts_dataset, i, n_few_shot, mode, random_fewshot, justification)

    messages = [{"role": "system", "content": system_prompt}]
    for ex in few_shots:
        user_content = f"Text: {ex['text']}\n" if within_context else f"Context: {ex['context']}\nText: {ex['text']}\n"
        if mode == "words":
            assistant_content = '{\n  "emphasized_word": ' + str(ex['emphasized_output']) + '\n}'
            if justification and ex['justification']:
                assistant_content = (
                    '{\n  "emphasized_word": ' + str(ex['emphasized_output']) + ',\n'
                    + '  "justification": "' + ex['justification'] + '"\n}'
                )
        elif mode == "indices":
            assistant_content = '{\n  "emphasized_index": ' + str(ex['emphasized_output']) + '\n}'
        elif mode == "markup":
            assistant_content = ex['emphasized_output']
            if justification and ex['justification']:
                assistant_content = f"MARKED_TEXT: {ex['emphasized_output']}\nJUSTIFICATION: {ex['justification']}"
        else:
            assistant_content = str(ex['emphasized_output'])
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})

    if within_context:
        prompt_text = f"Text: {transcription}\n"
    else:
        prompt_text = f"Context: {sample['context']}\nText: {transcription}\n"
    messages.append({"role": "user", "content": prompt_text})

    if hasattr(text_tokenizer, 'apply_chat_template'):
        return text_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking_model
        )
    formatted = ""
    for m in messages:
        role = m["role"].capitalize()
        formatted += f"{role}: {m['content']}\n\n"
    return formatted + "Assistant:"


def get_response(model, tokenizer, model_type: str, formatted_prompt: str, inputs: dict,
                 thinking_model: bool = False) -> str:
    with torch.no_grad():
        generate_cfg = {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "min_p": 0,
            "top_k": 20,
            "max_new_tokens": 100 if not thinking_model else 32768,
        }
        if thinking_model:
            generate_cfg = {
                "max_new_tokens": 32768,
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "min_p": 0,
                "top_k": 20,
            }

        outputs = model.generate(**inputs, **generate_cfg)
        idx = inputs['input_ids'].shape[1]
        if thinking_model:
            try:
                idx = outputs[0].tolist().index(151668) + 1
            except ValueError:
                pass

        return tokenizer.decode(outputs[0][idx:], skip_special_tokens=True).strip("\n")


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _warn_string_field(key_name):
    print(f"[Warning] '{key_name}' is a string, converting to single-item list")


def parse_markup_text(raw: str, original_text: str) -> tuple:
    text = raw.strip().strip('"').strip("'").strip()
    justification = None

    if "MARKED_TEXT:" in text and "JUSTIFICATION:" in text:
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith("MARKED_TEXT:"):
                text = line[len("MARKED_TEXT:"):].strip()
            elif line.startswith("JUSTIFICATION:"):
                justification = line[len("JUSTIFICATION:"):].strip()

    emphasized_words = [w.strip() for w in re.findall(r'\*\*([^*]+)\*\*', text) if w.strip()]
    return emphasized_words, justification


def parse_with_literal_eval(raw: str, mode: str = "words") -> tuple:
    m = re.search(r'\{.*?\}', raw, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON-like block found in response: {raw!r}")
    try:
        data = ast.literal_eval(m.group(0))
    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Failed to parse block with literal_eval: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict, got {type(data).__name__}")

    if mode == "words":
        key_name, expected_type = "emphasized_word", str
    elif mode == "indices":
        key_name, expected_type = "emphasized_index", int
    elif mode == "markup":
        raise ValueError("Use parse_markup_text for markup mode")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if key_name not in data:
        raise ValueError(f"Missing '{key_name}' key in parsed data: {data}")

    val = data[key_name]
    if not isinstance(val, list):
        if isinstance(val, str):
            _warn_string_field(key_name)
            val = [val]
        else:
            raise ValueError(f"'{key_name}' must be a list, got {type(val).__name__}")

    for item in val:
        if not isinstance(item, expected_type):
            raise ValueError(f"{mode} items must be {expected_type.__name__}, got {type(item).__name__}")

    justification = data.get("justification")
    return val, justification


def _is_match(pred_norm: str, text_norm: str) -> bool:
    return bool(pred_norm and text_norm and ((pred_norm in text_norm) or (text_norm in pred_norm)))


def _get_f(mode: str, transcription: str, predicted_output):
    import numpy as np
    transcription_norm = normalizer(transcription)
    words = transcription_norm.split()
    predicted_binary = [0] * len(words)

    if mode in ("words", "markup"):
        valid = []
        for pw in predicted_output:
            pn = normalizer(pw)
            if any(_is_match(pn, w) for w in words):
                valid.append(pw)
            else:
                logging.warning(f"Predicted word '{pw}' not found in text: '{transcription_norm}'")
        pred_norm_list = [normalizer(w) for w in valid]
        for idx, w in enumerate(words):
            if any(_is_match(pn, w) for pn in pred_norm_list):
                predicted_binary[idx] = 1
        return predicted_binary, valid
    elif mode == "indices":
        valid_idx = [p for p in predicted_output if 0 <= p < len(words)]
        for idx in valid_idx:
            predicted_binary[idx] = 1
        return predicted_binary, [words[i] for i in valid_idx]
    else:
        return predicted_binary, []


def parse_response(response: str, transcription: str, mode: str = "words") -> tuple:
    if mode == "markup":
        predicted_output, _ = parse_markup_text(response, transcription)
    elif mode == "parsed":
        predicted_output = response
        mode = "words"
    else:
        predicted_output, _ = parse_with_literal_eval(response, mode)
    return _get_f(mode, transcription, predicted_output)
