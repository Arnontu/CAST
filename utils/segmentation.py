"""Audio segmentation utilities using Whisper word-level timestamps."""
import numpy as np
from whisper_normalizer.english import EnglishTextNormalizer

normalizer = EnglishTextNormalizer()


def edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            current_row.append(min(
                previous_row[j + 1] + 1,
                current_row[j] + 1,
                previous_row[j] + (c1 != c2),
            ))
        previous_row = current_row
    return previous_row[-1]


def find_subsequence_fuzzy(haystack: list, needle: list, threshold: float = 0.7):
    """Find needle in haystack with fuzzy word matching. Returns start index or None."""
    if not needle or not haystack:
        return None
    best_score, best_idx = 0.0, None
    for i in range(len(haystack) - len(needle) + 1):
        matches = sum(
            1 if haystack[i + j] == needle[j] else (0.5 if edit_distance(haystack[i + j], needle[j]) <= 2 else 0)
            for j in range(len(needle))
        )
        score = matches / len(needle)
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx if best_score >= threshold else None


def prepare_text_context_mode(sentence: str, context: str, separator: str, max_context_length: int):
    """
    Prepare text for context-concat mode: context + separator + sentence.

    Returns:
        (full_text, actual_context, needs_segmentation)
    """
    clean_sentence = sentence.replace("**", "").replace("*", "")
    if not context:
        return clean_sentence, "", False

    if len(context) > max_context_length:
        context = context[:max_context_length].rsplit(" ", 1)[0] + "..."

    full_text = f"{context}{separator}{clean_sentence}"
    if len(full_text) > 3000:
        remaining = 3000 - len(clean_sentence) - len(separator)
        if remaining > 100:
            context = context[:remaining].rsplit(" ", 1)[0] + "..."
            full_text = f"{context}{separator}{clean_sentence}"
        else:
            full_text = clean_sentence
            context = ""

    return full_text, context, bool(context)


def prepare_text_explicit_mode(sentence: str, stressed_index: int, pause_token: str = "[pause:0.25s]") -> str:
    """
    Prepare text for explicit mode: insert pause_token before the stressed word
    so the TTS model pauses to emphasise it.
    """
    clean = sentence.replace("**", "").replace("*", "")
    import re
    tokens = re.findall(r"\S+", clean)
    if pause_token and 0 <= stressed_index < len(tokens):
        tokens.insert(stressed_index, pause_token)
    return " ".join(tokens)


def segment_text_portion(
    wav: np.ndarray,
    sample_rate: int,
    target_text: str,
    full_transcription: str,
    chunks: list,
    context: str = "",
    separator: str = " ",
):
    """
    Segment audio to extract only the portion corresponding to target_text.

    Uses Whisper word-level timestamps. Falls back gracefully when alignment fails.

    Returns:
        Segmented waveform (np.ndarray) or None if segmentation fails.
    """
    target_norm = normalizer(target_text)
    full_norm = normalizer(full_transcription)
    target_words = target_norm.split()
    full_words = full_norm.split()

    if not target_words or not full_words:
        return None

    match_start_idx = find_subsequence_fuzzy(full_words, target_words, threshold=0.7)

    if match_start_idx is None:
        if target_norm in full_norm:
            before_target = full_norm.split(target_norm)[0]
            match_start_idx = len(before_target.split())
        elif context:
            context_words = normalizer(context).split()
            match_start_idx = min(len(context_words), len(full_words) - len(target_words))
        else:
            return None

    if not chunks or match_start_idx >= len(chunks):
        return None

    end_word_idx = min(match_start_idx + len(target_words), len(chunks))

    try:
        if isinstance(chunks[0], dict):
            start_time = chunks[match_start_idx]["timestamp"][0]
            end_time = chunks[end_word_idx - 1]["timestamp"][1]
        else:
            start_time = chunks[match_start_idx].start
            end_time = chunks[end_word_idx - 1].end
    except (KeyError, IndexError, AttributeError) as e:
        print(f"Warning: Could not extract timestamps: {e}")
        return None

    if start_time is None:
        start_time = 0.0
    if end_time is not None and end_time <= start_time:
        return None
    if end_time is None:
        end_time = len(wav) / sample_rate

    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    if start_sample < 0 or end_sample > len(wav) or start_sample >= end_sample:
        return None

    padding = int(0.05 * sample_rate)
    return wav[max(0, start_sample - padding): min(len(wav), end_sample + padding)]
