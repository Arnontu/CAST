#!/usr/bin/env python3
"""LLM pipeline to generate and validate contrastive word-stress sentence pairs for CAST."""

from __future__ import annotations
import random
import jinja2
import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import random
import jsonlines
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI, APIError, RateLimitError
from anthropic import Anthropic

import backoff
from datetime import datetime
from threading import Lock
_JSONL_WRITE_LOCK = Lock()
POS_PLAN = "SEMD"
FN_GEN_SYS = "gen.system.j2"
FN_GEN_USR = "gen.user.j2"
FN_JUDGE_SYS = "judge.system.j2"
FN_JUDGE_USR = "judge.user.j2"

# -----------------------------
# Models (kept minimal)
# -----------------------------

class Item(BaseModel):
    sentence: str = Field(..., alias="s")
    context: str = Field(..., alias="c")
    stressed_word: str = Field(..., alias="w")
    judge_match: Optional[bool] = None
    topic: Optional[str] = None
    notes: Optional[str] = None
    mask: List[int] = Field(default_factory=list)
    stressed_index: Optional[int] = None
    item_id: Optional[str] = None
    gen_model: Optional[str] = None
    pos_code: Optional[str] = None   # one of S,E,M,D for this item
    mode_code: Optional[str] = None  # one of S,Q,C,R,X for this item
    ctx_code: Optional[str] = None   # one of E,C,A,L,T,R,Q,M,P for this item
    gen_style: Optional[str] = None      # one of Polite and formal (customer service style), Casual and direct (friends chatting), Urgent and concise (emergency or rush context), Explanatory and patient (teacher style) for this item
    judge_word: Optional[str] = None
    judge_confidence: Optional[float] = None
    judge_cue: Optional[str] = None
    judge_trigger: Optional[str] = None
    judge_counter_context: Optional[str] = None
    judge_counter_word: Optional[str] = None
    judge_counter_cue: Optional[str] = None
    judge_counter_trigger: Optional[str] = None
    judge_flip: Optional[bool] = None
    judge_strong_flip: Optional[bool] = None  
    judge_cc_quality_ok: Optional[bool] = None
    run_id: Optional[str] = None
    step: Optional[str] = None

    
    group_id: Optional[str] = None

    model_config = {
        "populate_by_name": True,  # allows either field names or aliases
        "extra": "ignore",
    }


class BatchItems(BaseModel):
    items: List[Item]

# -----------------------------
# Small helpers
# -----------------------------
import json
import re
from typing import Optional

def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        # remove opening fence (``` or ```json etc)
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        # remove closing fence if present
        t = re.sub(r"\s*```$", "", t).strip()
    return t

def extract_json_object(text: str) -> str:
    t = _strip_code_fences(text)

    # direct parse
    json.loads(t)
    return t

def looks_like_truncation(text: str) -> bool:
    if not text or not text.strip():
        return True
    t = text.strip()
    # common truncation signals
    if t.startswith("```") and not t.rstrip().endswith("```"):
        return True
    if t.count("{") > t.count("}"):
        return True
    if t.count("[") > t.count("]"):
        return True
    if t.rstrip().endswith((",", ":", "\"")):
        return True
    return False

def salvage_truncated_results_json(text: str) -> Optional[str]:
    """
    Attempt to salvage your specific schema:
      { "results": [ {...}, {...}, ... ] }

    Strategy:
    - strip code fences
    - find `"results"` array opening
    - keep up to the last complete object end `}` inside the array
    - close the array `]` and outer object `}`
    - validate with json.loads; return salvaged JSON string on success
    """
    t = _strip_code_fences(text)
    if not t:
        return None

    # Find the start of the results array
    m = re.search(r'"\s*results\s*"\s*:\s*\[', t)
    if not m:
        return None

    prefix_end = m.end()  # points just after the '['
    prefix = t[:prefix_end]

    # Work on the substring after '[' to find last complete object boundary
    after = t[prefix_end:]

    # Find last '}' in the "after" region
    last_obj_end = after.rfind("}")
    if last_obj_end == -1:
        # no complete object at all
        return None

    body = after[: last_obj_end + 1].rstrip()

    # Remove trailing comma after the last complete object, if present
    body = re.sub(r",\s*$", "", body)

    # Construct a closed JSON
    candidate = prefix + body + "]}"
    # Sometimes there is extra text before "results"; ensure outer object opened
    # If prefix doesn't contain a '{', try to wrap.
    if "{" not in candidate:
        candidate = "{" + candidate

    # Validate
    try:
        json.loads(candidate)
        return candidate
    except Exception:
        return None

def extract_or_salvage_json(text: str) -> str:
    """
    1) Try strict extraction (must be valid JSON)
    2) If fails and looks like truncation, try salvage
    3) Else raise
    """
    try:
        return extract_json_object(text)
    except Exception as e:
        if looks_like_truncation(text):
            salvaged = salvage_truncated_results_json(text)
            if salvaged is not None:
                return salvaged
        raise



def warn_on_low_diversity(rows):
    import re
    n = len(rows)
    if n == 0: return
    pat = re.compile(r"^[A-Z][a-z]+ the \*\*[^*]+?\*\*")
    tmpl = sum(1 for r in rows if pat.search(r.get("sentence","")))
    pos2 = sum(1 for r in rows if r.get("stressed_index") == 2)
    questions = sum(1 for r in rows if r.get("sentence","").strip().endswith("?"))
    if tmpl / n > 0.2:
        print(f"[warn] {tmpl}/{n} items start with 'Verb the **NOUN**' frame.")
    if pos2 / n > 0.3:
        print(f"[warn] {pos2}/{n} items stress at index 2.")
    if questions / n < 0.15:
        print(f"[warn] only {questions}/{n} questions; mode mix may be narrow.")

def sha8(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")

def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    n = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(path, "a") as w:
        for r in rows:
            w.write(r)
            n += 1
    return n

def read_all_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with jsonlines.open(path, "r") as r:
        for obj in r:
            out.append(obj)
    return out

def now_ts() -> int:
    return int(time.time())

def write_flip_positive(src: Path, dst: Path) -> Tuple[int, int]:
    """Write only items where judge_flip is True. Returns (kept, total)."""
    kept = 0
    total = 0
    with jsonlines.open(dst, "w") as w, jsonlines.open(src, "r") as r:
        for obj in r:
            total += 1
            jf = bool(obj.get("judge_flip"))
            if jf:
                w.write(obj)
                kept += 1
    return kept, total

def _ban_slices_for_wave(big_ban: List[str], wave_size: int) -> List[List[str]]:
    """Partition BAN across concurrent requests so each gets a different slice."""
    if not big_ban or wave_size <= 0:
        return [[] for _ in range(max(1, wave_size))]
    slice_size = max(8, min(16, math.ceil(len(big_ban) / wave_size)))
    pool = big_ban[:]  # copy
    random.shuffle(pool)
    slices = [[] for _ in range(wave_size)]
    j = 0
    for term in pool:
        if len(slices[j]) < slice_size:
            slices[j].append(term)
        j = (j + 1) % wave_size
    return slices


def _ban_slices_for_wave_with_size(big_ban: List[str], wave_size: int, slice_size: int) -> List[List[str]]:
    """Partition BAN across concurrent requests with explicit slice size.

    Args:
        big_ban: Full list of banned terms
        wave_size: Number of concurrent requests
        slice_size: Size of each slice. If 0, all words are used (no slicing).
    """
    if not big_ban or wave_size <= 0:
        return [[] for _ in range(max(1, wave_size))]

    # If slice_size is 0, return all words for each wave (no slicing)
    if slice_size == 0:
        return [big_ban[:] for _ in range(wave_size)]

    # Use provided slice_size
    pool = big_ban[:]  # copy
    random.shuffle(pool)
    slices = [[] for _ in range(wave_size)]
    j = 0
    for term in pool:
        if len(slices[j]) < slice_size:
            slices[j].append(term)
        j = (j + 1) % wave_size
    return slices

def write_flip_strong(src: Path, dst: Path, min_conf: float = 0.6) -> Tuple[int, int]:
    """Write only items with a valid flip and a high quality counter_context."""
    kept = 0
    total = 0
    with jsonlines.open(dst, "w") as w, jsonlines.open(src, "r") as r:
        for obj in r:
            total += 1
            if not obj.get("judge_flip"):
                continue
            if not obj.get("judge_cc_quality_ok"):
                continue
            # if float(obj.get("judge_confidence") or 0.0) < min_conf:
            #     continue
            sw = (obj.get("stressed_word") or "").lower()
            cc = (obj.get("judge_counter_context") or "").lower()
            if sw and cc and sw in cc:
                continue
            w.write(obj)
            kept += 1
    return kept, total
def get_key(model):
    import os
    if 'gemini' in model:
        key = os.environ.get("GOOGLE_API_KEY")
    elif 'gpt' in model:
        key = os.environ.get("OPENAI_API_KEY")
    elif 'claude' in model:
        key = os.environ.get("ANTHROPIC_API_KEY")
    else:
        raise ValueError(f"Unknown model for key retrieval: {model}")
    if not key:
        var = "GOOGLE_API_KEY" if 'gemini' in model else ("ANTHROPIC_API_KEY" if 'claude' in model else "OPENAI_API_KEY")
        raise EnvironmentError(f"API key not set. Export {var} before running.")
    return key

def get_client(model: str) -> OpenAI:
    key = get_key(model)
    if 'gemini' in model:
        client = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/",api_key=key)
    elif 'claude' in model:
        client = Anthropic(api_key=key)
    else:
        client = OpenAI(api_key=key)
    return client

def get_prompts_paths(args) -> Tuple[Path, Path, Path, Path]:
    if args.prompt_dir:
        pd = args.prompt_dir
        if not pd.exists():
            sys.exit(f"Error: Prompt directory not found: {pd}")
            
        gen_sys_p = pd / FN_GEN_SYS
        gen_usr_p = pd / FN_GEN_USR
        judge_sys_p = pd / FN_JUDGE_SYS
        judge_usr_p = pd / FN_JUDGE_USR
        
        # Optional: Warn if individual files are missing but dir exists
        for p in [gen_sys_p, gen_usr_p, judge_sys_p, judge_usr_p]:
            if not p.exists():
                raise("Prompt file missing: {p.name}")
        return gen_sys_p, gen_usr_p, judge_sys_p, judge_usr_p
    else:
        raise ValueError("Prompt directory must be provided via --prompt-dir argument.")

def _save_prompt_txt(out_txt: Path, *, name: str, model: str, run_id: str,
                     system: str, user: str, extra: dict | None = None, append: bool = False) -> None:
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = [
        f"=== {name} PROMPT ===",
        f"time: {stamp}",
        f"run_id: {run_id}",
        f"model: {model}",
    ]
    if extra:
        # keep it compact but readable
        header.append("extra: " + json.dumps(extra, ensure_ascii=False))
    header_line = "\n".join(header)
    body = f"{header_line}\n--- system ---\n{system}\n--- user ---\n{user}\n\n"
    mode = "a" if append else "w"
    with open(out_txt, mode, encoding="utf-8") as f:
        f.write(body)





# -----------------------------
# Standard continue helpers
# -----------------------------
def _row_key_triple(sentence: str, context: str, stressed_word: str) -> str:
    """Stable key for an item by its core fields."""
    payload = json.dumps([sentence or "", context or "", stressed_word or ""], separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()

def _jsonl_count_valid_items(path: Path) -> int:
    """Count rows that parse into Item in a jsonl file."""
    if not path.exists():
        return 0
    n = 0
    with jsonlines.open(path, "r") as r:
        for obj in r:
            try:
                Item.model_validate(obj)
                n += 1
            except ValidationError:
                continue
    return n

def _jsonl_item_keys(path: Path) -> set[str]:
    """Collect item keys from a jsonl with either gen rows or valid rows."""
    keys = set()
    if not path.exists():
        return keys
    with jsonlines.open(path, "r") as r:
        for obj in r:
            # rows may be gen or valid shape
            s = obj.get("sentence") or obj.get("s") or ""
            c = obj.get("context") or obj.get("c") or ""
            w = obj.get("stressed_word") or obj.get("w") or ""
            if s and w:
                keys.add(_row_key_triple(s, c, w))
    return keys

def _filter_unvalidated_items(gen_items: list[Item], valid_keys: set[str]) -> list[Item]:
    out = []
    for it in gen_items:
        k = _row_key_triple(it.sentence, it.context, it.stressed_word)
        if k not in valid_keys:
            out.append(it)
    return out

# -----------------------------
# Enrichment and cleanup
# -----------------------------
import re, string

_STRESS_RE = re.compile(r"\*\*(\w[\w'-]*?)\*\*")

def _strip_punct(tok: str) -> str:
    # keep common interior apostrophes and hyphens
    return tok.strip().strip("“”‘’'\"").strip(".,!?;:()[]{}")

def enrich_and_clean_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    - prefer stressed_word from **…** if present
    - compute stressed_index and mask
    - drop empty fields: None, "", []
    """
    sentence = row.get("sentence", "")
    m = _STRESS_RE.search(sentence)
    stressed_from_markup = m.group(1) if m else None
    if stressed_from_markup:
        row["stressed_word"] = stressed_from_markup

    clean_sentence = sentence.replace("**", "")
    toks = clean_sentence.split()
    norm = [_strip_punct(t).lower() for t in toks]
    sw = (row.get("stressed_word") or "").strip()
    sw_norm = _strip_punct(sw).lower() if sw else None

    stressed_index = None
    if sw_norm:
        for i, nt in enumerate(norm):
            if nt == sw_norm:
                stressed_index = i
                break

    if stressed_index is not None and toks:
        row["stressed_index"] = stressed_index
        mask = [0] * len(toks)
        mask[stressed_index] = 1
        row["mask"] = mask
    
    # remove markup, * _ from word within the context as well
    context = row.get("context", "")
    #not like this since this not in a word level: clean_context = context.replace("**", "").replace("*", "").replace("_", "")
    if context:
        clean_context = re.sub(r"[\*_]{1,2}(\w[\w'-]*?)[\*_]{1,2}", r"\1", context)
        row["context"] = clean_context

    # drop empties
    return {
        k: v for k, v in row.items()
        if v is not None and v != "" and not (isinstance(v, list) and len(v) == 0)
    }

# -----------------------------
# Prompts
# -----------------------------
import re
from collections import Counter

def _head_form(w: str) -> str:
    w = (w or "").strip().lower()
    if not w:
        return ""
    # very light singularization
    if w.endswith("ies") and len(w) > 3:
        return w[:-3] + "y"
    if re.search(r"(ses|xes|zes|ches|shes)$", w):
        return w[:-2]
    if w.endswith("s") and len(w) > 3:
        return w[:-1]
    return w

def build_ban_from_jsonl(paths, top_k: int = 20) -> list[str]:
    """Collect most frequent stressed heads from one or more jsonl files."""
    cnt = Counter()
    for p in paths:
        if not Path(p).exists():
            continue
        with jsonlines.open(p, "r") as r:
            for obj in r:
                h = _head_form(obj.get("stressed_word", ""))
                if h:
                    cnt[h] += 1
    return [w for w, _ in cnt.most_common(top_k)]

def load_ban_words_from_sources(sources: List[str], top_k: int = 0) -> list[str]:
    """
    Load banned words from multiple sources (JSONL files or HF datasets).

    Args:
        sources: List of paths to JSONL files or HF dataset identifiers
        top_k: If > 0, return only top_k most frequent words. If 0, return all.

    Returns:
        List of banned word forms
    """
    cnt = Counter()

    for src in sources:
        src_path = Path(src)

        # Check if it's a local JSONL file
        if src_path.exists() and src_path.suffix in ('.jsonl', '.json'):
            with jsonlines.open(src_path, "r") as r:
                for obj in r:
                    h = _head_form(obj.get("stressed_word", ""))
                    if h:
                        cnt[h] += 1
        else:
            # Assume it's an HF dataset identifier
            try:
                from datasets import load_from_disk
                ds = load_from_disk(src)
                for obj in ds:
                    h = _head_form(obj.get("stressed_word", ""))
                    if h:
                        cnt[h] += 1
            except Exception as e:
                print(f"[warn] Could not load '{src}' as HF dataset: {e}")
                continue

    if top_k > 0:
        return [w for w, _ in cnt.most_common(top_k)]
    else:
        return list(cnt.keys())
# helpers, near _make_len_plan
def _make_ctx_plan(batch_size: int) -> str:
    # E=exclusivity, C=correction, A=alternative, L=location, T=timing,
    # R=role/permission, Q=quantity, M=modality, P=pronoun-contrast
    base = "ECALTRQMP"
    plan = (base * ((batch_size + len(base) - 1) // len(base)))[:batch_size]
    import random
    lst = list(plan)
    random.shuffle(lst)
    return "".join(lst)

def _normalize_items_object(obj: dict) -> dict:
    """Accept either dict items or compact list triples and normalize to dict items."""
    items = obj.get("items", [])
    norm = []
    for it in items:
        if isinstance(it, dict):
            # already in {"sentence":...,"context":...,"stressed_word":...} (or s/c/w) form
            if "sentence" in it or "s" in it:
                norm.append(it)
        elif isinstance(it, (list, tuple)) and len(it) >= 3:
            s, c, w = it[0], it[1], it[2]
            norm.append({"sentence": s, "context": c, "stressed_word": w})
        # else: skip malformed rows silently
    return {"items": norm}


def parse_response_to_rows(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parses LLM output (either new 'groups' format or legacy 'items' format)
    and flattens it into a list of row dictionaries.
    """
    rows = []
    
    # --- New Format: Groups ---
    if "groups" in obj:
        for g_idx, group in enumerate(obj["groups"]):
            sent = group.get("sentence", "").strip()
            # Create a stable ID for the group based on the sentence
            gid = sha8(sent)
            
            variations = group.get("variations", [])
            for v_idx, var in enumerate(variations):
                rows.append({
                    "sentence": sent,
                    "context": var.get("context"),
                    "stressed_word": var.get("stressed_word"),
                    "group_id": gid,
                    # Metadata to help mapping plans later
                    "_is_primary": (v_idx == 0),
                    "_group_index": g_idx
                })

    # --- Legacy Format: Items ---
    elif "items" in obj:
        # Normalize list/tuple items to dict
        raw_items = obj["items"]
        for idx, it in enumerate(raw_items):
            row = {}
            if isinstance(it, dict):
                row = it
            elif isinstance(it, (list, tuple)) and len(it) >= 3:
                row = {"sentence": it[0], "context": it[1], "stressed_word": it[2]}
            
            # Legacy items are treated as independent groups of 1
            if "sentence" in row:
                row["group_id"] = sha8(row["sentence"].strip())
                row["_is_primary"] = True
                row["_group_index"] = idx
                rows.append(row)
                
    return rows
def _make_len_plan(batch_size: int, min_len: int, max_len: int) -> str:
    if min_len > max_len:
        min_len, max_len = max_len, min_len
    span = max_len - min_len
    num_anchors = min(7, span + 1) or 1
    if num_anchors == 1:
        anchors = [min_len]
    else:
        step = span / (num_anchors - 1)
        anchors = [int(round(min_len + i * step)) for i in range(num_anchors)]
    seq = (anchors * ((batch_size + len(anchors) - 1) // len(anchors)))[:batch_size]
    random.shuffle(seq)
    jitter_choices = [-1, 0, 1] if span >= 2 else [0]
    out = []
    for v in seq:
        v2 = v + random.choice(jitter_choices)
        v2 = max(min_len, min(max_len, v2))
        out.append(v2)
    return ",".join(map(str, out))


def build_generator_prompt(
    batch_size: int,
    topics: Optional[List[str]],
    min_len: int,
    max_len: int,
    pos_plan: Optional[str] = None,
    mode_plan: Optional[str] = None,
    micro_cues: Optional[List[str]] = None,
    banned_terms: Optional[List[str]] = None,
    ctx_plan: Optional[str] = None,
    context_ban_terms: Optional[List[str]] = None,
    styles: Optional[List[str]] = None,
    # NEW ARGUMENTS
    system_path: Optional[Path] = None,
    user_path: Optional[Path] = None,
) -> Tuple[str, str, Optional[str], Optional[str], Optional[str], Optional[str]]:
# --- 1. Python Logic (Calculate Plans & Variables) ---
    span = max_len - min_len
    target_len = random.randint(min_len + 2, max_len - 2) if span >= 4 else (min_len + max_len) // 2

    if not pos_plan:
        pos_plan = (POS_PLAN * ((batch_size + len(POS_PLAN)-1) // len(POS_PLAN)))[:batch_size]
    if not mode_plan:
        mode_plan = ("SQCRX" * ((batch_size + 4) // 5))[:batch_size]

    # Pre-calculate legacy hint strings (useful if you want simple templates)
    topic_hint = ", ".join(t.strip() for t in topics) + ". " if topics else ""
    micro_hint = ",".join(micro_cues[:8]) + " (guide only)." if micro_cues else ""
    ban_hint = ", ".join(sorted(set(banned_terms))) + ". " if banned_terms else "None"
    ctx_hint = f"CTX_PLAN={ctx_plan}  " if ctx_plan else ""
    ctx_ban_hint =  ", ".join(sorted(set(context_ban_terms))) + ". " if context_ban_terms else ""

    if not styles:
        styles = [
            "Polite and formal",
            "Casual and direct",
            "Urgent and concise",
            "Explanatory and patient"
        ]
        # styles = [
        #     "Polite and formal (customer service style)",
        #     "Casual and direct (friends chatting)",
        #     "Urgent and concise (emergency or rush context)",
        #     "Explanatory and patient (teacher style)"
        # ]
    selected_style = random.choice(styles)

    # --- 2. Render Logic ---
    
    # Context available to Jinja
    ctx = {
        "batch_size": batch_size,
        "min_len": min_len,
        "max_len": max_len,
        "target_len": target_len,
        # Raw lists (for looping in Jinja)
        "topics": topics or [],
        "micro_cues": micro_cues or [],
        "banned_terms": banned_terms or [],
        "context_ban_terms": context_ban_terms or [],
        # Plans
        "pos_plan": pos_plan,
        "mode_plan": mode_plan,
        "ctx_plan": ctx_plan,
        "selected_style": selected_style,
        # Pre-formatted strings (for easy insertion)
        "topic_hint": topic_hint,
        "micro_hint": micro_hint,
        "ban_hint": ban_hint,
        "ctx_hint": ctx_hint,
        "ctx_ban_hint": ctx_ban_hint,
    }

    if system_path and user_path and system_path.exists() and user_path.exists():
        # Load and Render with Jinja2
        sys_t = jinja2.Template(system_path.read_text(encoding="utf-8"))
        usr_t = jinja2.Template(user_path.read_text(encoding="utf-8"))
        
        system = sys_t.render(**ctx)
        user = usr_t.render(**ctx)
    else:
        raise ValueError("Both system_path and user_path must be provided and exist.")
    # else:
    #     # --- Fallback to Hardcoded Logic ?---


    return system, user, pos_plan, mode_plan, ctx_plan, selected_style



def build_judge_batch_prompt(items: List[Item],
                             system_path: Optional[Path] = None, 
    user_path: Optional[Path] = None) -> Tuple[str, str]:
    # Ensure confidence instruction uses a placeholder/example that isn't 0.0
    if system_path and user_path and system_path.exists() and user_path.exists():
        ctx = {
            "items": items,
            "count": len(items)
        }
        
        sys_t = jinja2.Template(system_path.read_text(encoding="utf-8"))
        usr_t = jinja2.Template(user_path.read_text(encoding="utf-8"))
        
        system = sys_t.render(**ctx)
        user = usr_t.render(**ctx)
    else:
        raise ValueError("Both system_path and user_path must be provided and exist.")


    return system, user

# -----------------------------
# Standard sync API helpers
# -----------------------------

def _chat_json(client, model: str, system: str, user: str, temperature: float = 0.5) -> str:
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


@backoff.on_exception(backoff.expo, (APIError, RateLimitError, TimeoutError), max_tries=5, jitter=backoff.full_jitter)
def _standard_gen_once(client: OpenAI, model: str, k: int, topics: List[str], micro_cues: List[str],
                       min_len: int, max_len: int, pos_plan: str, mode_plan: str,
                       out_gen_jsonl: Path, run_id: str,
                       banned_terms: Optional[List[str]] = None,
                       ctx_plan: Optional[str] = None,
                       context_ban_terms: Optional[List[str]] = None,
                       system_path: Optional[Path] = None,
                       user_path: Optional[Path] = None,
                       ) -> int:
    system, user, pos_plan_used, mode_plan_used, ctx_plan_used, selected_style = build_generator_prompt(
        batch_size=k,
        topics=topics or [],
        min_len=min_len,
        max_len=max_len,
        pos_plan=pos_plan,
        mode_plan=mode_plan,
        micro_cues=micro_cues or [],
        banned_terms=banned_terms or [],
        ctx_plan=ctx_plan,
        context_ban_terms=context_ban_terms or [],
        system_path=system_path,
        user_path=user_path,
    )
    txt = _chat_json(client, model, system, user)
    try:
        obj = json.loads(txt)
        raw_rows = parse_response_to_rows(obj)
    except Exception as e:
        print(f"[warn] JSON parse failed: {e}")
        return 0
    rows = []
    for row in raw_rows:
        try:
            it = Item.model_validate(row)
            dump = it.model_dump()
            g_idx = row.get("_group_index", -1)
            is_primary = row.get("_is_primary", False)
            if is_primary and g_idx >= 0:
                if pos_plan_used and g_idx < len(pos_plan_used):
                    dump["pos_code"] = pos_plan_used[g_idx]
                if mode_plan_used and g_idx < len(mode_plan_used):
                    dump["mode_code"] = mode_plan_used[g_idx]
                if ctx_plan_used and g_idx < len(ctx_plan_used):
                    dump["ctx_code"] = ctx_plan_used[g_idx]
            if selected_style:
                dump["gen_style"] = selected_style
            dump["run_id"] = run_id
            dump["step"] = "generated"
            dump["gen_model"] = model
            s = dump.get("sentence") or ""
            c = dump.get("context") or ""
            w = dump.get("stressed_word") or ""
            dump["item_id"] = _row_key_triple(s, c, w)
            dump = enrich_and_clean_row(dump)
            rows.append(dump)
        except ValidationError:
            continue
    warn_on_low_diversity(rows)
    with _JSONL_WRITE_LOCK:
        gen_prompt_txt_path = Path(out_gen_jsonl).with_name("gen.prompt.txt")
        if gen_prompt_txt_path.exists() and gen_prompt_txt_path.stat().st_size == 0:
            _save_prompt_txt(gen_prompt_txt_path, name="gen", model=model, run_id=run_id, system=system, user=user)
        elif not gen_prompt_txt_path.exists():
            gen_prompt_txt_path.write_text("")
        return append_jsonl(out_gen_jsonl, rows)


_TIME_PAT = re.compile(
    r"\b(?:mon|tue|wed|thu|fri|sat|sun|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"today|tomorrow|yesterday|\d{1,2}(?::\d{2})?(?:am|pm)?)\b",
    re.I,
)


@backoff.on_exception(backoff.expo, (APIError, RateLimitError, TimeoutError), max_tries=5, jitter=backoff.full_jitter)
def _standard_judge_once(client: OpenAI, model: str, items_chunk: List[Item],
                         out_valid_jsonl: Path, run_id: str,
                         system_path: Optional[Path] = None,
                         user_path: Optional[Path] = None) -> int:
    def _sent_tokens(s: str) -> List[str]:
        clean = (s or "").replace("**", "")
        return [_strip_punct(t).lower() for t in clean.split() if t.strip()]

    system, user = build_judge_batch_prompt(items_chunk, system_path=system_path, user_path=user_path)
    txt = _chat_json(client, model, system, user)
    obj = json.loads(txt)
    try:
        results = obj.get("results", [])
    except Exception as e:
        print(f"[warn] invalid JSON: {e}")
        results = []

    rows = []
    for base, r in zip(items_chunk, results):
        jw   = (r.get("stressed_word") or "").strip().lower() if isinstance(r, dict) else ""
        conf = float(r.get("confidence") or -1.0) if isinstance(r, dict) else 0.0
        cue  = (r.get("cue") or "").strip().upper() if isinstance(r, dict) else ""
        trig = (r.get("trigger") or "").strip() if isinstance(r, dict) else ""
        ccx  = (r.get("counter_context") or "").strip() if isinstance(r, dict) else ""
        cst  = (r.get("counter_stressed_word") or "").strip().lower() if isinstance(r, dict) else ""
        ccue = (r.get("counter_cue") or "").strip().upper() if isinstance(r, dict) else ""
        ctrg = (r.get("counter_trigger") or "").strip() if isinstance(r, dict) else ""

        toks = _sent_tokens(base.sentence)
        cst_present = bool(cst and cst in toks)
        flip_ok = bool(cst_present and jw and cst != jw)

        cc_toks = [_strip_punct(t).lower() for t in ccx.split() if t.strip()]
        len_ok = 0 < len(cc_toks) <= 10
        sw = (base.stressed_word or "").strip().lower()
        no_quote_target = bool(ccx) and (sw not in ccx.lower())
        no_markup = ("**" not in ccx)
        no_new_time = not _TIME_PAT.search(ccx) or cst in {"today", "tomorrow", "yesterday"}
        judge_abstained = (not cst) and (not ccx)
        cc_quality_ok = bool(
            judge_abstained
            or (len_ok and no_quote_target and no_markup and cst_present and no_new_time)
        )

        row = {
            **base.model_dump(),
            "judge_word": jw,
            "judge_match": (jw.lower() == (base.stressed_word or "").strip().lower()),
            "judge_confidence": conf,
            "judge_cue": cue,
            "judge_trigger": trig,
            "judge_counter_context": ccx,
            "judge_counter_word": cst,
            "judge_counter_cue": ccue,
            "judge_counter_trigger": ctrg,
            "judge_flip": flip_ok,
            "judge_cc_quality_ok": cc_quality_ok,
            "run_id": run_id,
            "step": "validated",
            "valid_model": model,
        }
        rows.append(enrich_and_clean_row(row))

    with _JSONL_WRITE_LOCK:
        valid_prompt_txt_path = Path(out_valid_jsonl).with_name("valid.prompt.txt")
        if valid_prompt_txt_path.exists() and valid_prompt_txt_path.stat().st_size == 0:
            _save_prompt_txt(valid_prompt_txt_path, name="valid", model=model, run_id=run_id, system=system, user=user)
        elif not valid_prompt_txt_path.exists():
            valid_prompt_txt_path.write_text("")
        return append_jsonl(out_valid_jsonl, rows)


# -----------------------------
# Commands
# -----------------------------

def _run_generation(args, gen_client, out_dir: Path, out_gen: Path, prompts: tuple) -> None:
    """Concurrent LLM generation phase: fills gen.jsonl up to args.gen_total."""
    gen_sys_p, gen_usr_p, _, _ = prompts
    already_gen = _jsonl_count_valid_items(out_gen)
    target_gen = max(0, args.gen_total)
    remaining = max(0, target_gen - already_gen)
    print(f"[gen] existing={already_gen}  target={target_gen}  remaining={remaining}")

    if remaining == 0:
        print("[gen] nothing to do. gen.jsonl already meets the target.")
        return

    if args.only_valid:
        print("[gen] skipped: --only-valid specified.")
        return

    k = args.gen_items_per_req
    n_reqs = (remaining + k - 1) // k
    wave_size = max(1, args.gen_concurrency)
    judge_model_safe = args.judge_model.replace("/", "_").replace(":", "_")
    out_valid = out_dir / f"valid_{judge_model_safe}.jsonl"

    user_ban_slices = [[] for _ in range(wave_size)]
    if args.ban_list:
        user_ban_words = load_ban_words_from_sources(args.ban_list, top_k=0)
        print(f"[ban] loaded {len(user_ban_words)} banned words from user sources")
        user_ban_slices = _ban_slices_for_wave_with_size(user_ban_words, wave_size, args.slice_size)

    for start in range(0, n_reqs, wave_size):
        end = min(start + wave_size, n_reqs)
        ban_sources = [p for p in [out_gen, out_valid] if p.exists()]
        dynamic_ban = build_ban_from_jsonl(ban_sources, top_k=60)
        dynamic_slices = _ban_slices_for_wave(dynamic_ban, wave_size)
        ban_slices = [list(set(user_ban_slices[i] + dynamic_slices[i])) for i in range(wave_size)]

        with ThreadPoolExecutor(max_workers=wave_size) as ex:
            futs = []
            for n_i, i in enumerate(range(start, end)):
                remaining_now = remaining - i * k
                this_k = min(k, remaining_now)
                pos_plan = (POS_PLAN * ((this_k + len(POS_PLAN) - 1) // len(POS_PLAN)))[:this_k]
                mode_plan = ("SQCRX" * ((this_k + 4) // 5))[:this_k]
                ctx_plan = _make_ctx_plan(this_k)
                ban_slice = ban_slices[n_i] if n_i < len(ban_slices) else []
                futs.append(ex.submit(
                    _standard_gen_once, gen_client, args.gen_model, this_k,
                    args.topics or [], args.micro_cues or [],
                    args.min_len, args.max_len, pos_plan, mode_plan,
                    out_gen, args.run_id or "",
                    banned_terms=ban_slice,
                    ctx_plan=ctx_plan,
                    context_ban_terms=["only"],
                    system_path=gen_sys_p,
                    user_path=gen_usr_p,
                ))
            for f in as_completed(futs):
                try:
                    print(f"[gen] wrote {f.result()} items")
                except Exception as e:
                    print(f"[err] a generation request failed: {e}")


def _run_validation(args, judge_client, out_dir: Path, out_gen: Path, out_valid: Path, prompts: tuple) -> None:
    """Validation phase: reads gen.jsonl, de-interleaves siblings, judges → valid_*.jsonl."""
    _, _, judge_sys_p, judge_usr_p = prompts

    if not out_gen.exists() or out_gen.stat().st_size == 0:
        print("[val] skipped: no gen.jsonl produced.")
        return

    src_items: list[Item] = []
    for obj in jsonlines.open(out_gen):
        try:
            src_items.append(Item.model_validate(obj))
        except ValidationError:
            continue
    if not src_items:
        print("[val] no valid items parsed from gen.jsonl")
        return

    if args.filter_banned and args.ban_list:
        ban_words_set = set(load_ban_words_from_sources(args.ban_list, top_k=0))
        before = len(src_items)
        src_items = [it for it in src_items if _head_form(it.stressed_word) not in ban_words_set]
        print(f"[val] filtered banned words: {before} -> {len(src_items)}")

    if args.validate_pairs:
        from collections import Counter
        from itertools import groupby as _groupby
        group_counts = Counter(it.group_id or it.sentence for it in src_items)
        valid_groups = {gid for gid, cnt in group_counts.items() if cnt >= 2}
        before = len(src_items)
        src_items = [it for it in src_items if (it.group_id or it.sentence) in valid_groups]
        print(f"[val] filtered for pairs: {before} -> {len(src_items)}")

    if not src_items:
        print("[val] no items remaining after filtering")
        return

    valid_keys = _jsonl_item_keys(out_valid)
    todo_items = _filter_unvalidated_items(src_items, valid_keys)
    print(f"[val] already_valid={len(src_items) - len(todo_items)}  to_validate={len(todo_items)}")

    if not todo_items:
        print("[val] nothing to validate. valid.jsonl is up to date.")
        return

    # De-interleave siblings so the same pair never lands in the same judge chunk
    from itertools import groupby as _groupby
    todo_items.sort(key=lambda x: x.group_id or x.sentence)
    list_a, list_b = [], []
    for _, group_iter in _groupby(todo_items, key=lambda x: x.group_id or x.sentence):
        g = list(group_iter)
        list_a.append(g[0])
        if len(g) > 1:
            list_b.append(g[1])
        if len(g) > 2:
            list_b.extend(g[2:])
    ordered = list_a + list_b

    chunks = [ordered[i: i + args.judge_chunk_size] for i in range(0, len(ordered), args.judge_chunk_size)]
    run_id = args.run_id or ""
    with ThreadPoolExecutor(max_workers=max(1, args.judge_concurrency)) as ex:
        futs = [
            ex.submit(_standard_judge_once, judge_client, args.judge_model, ch, out_valid, run_id, judge_sys_p, judge_usr_p)
            for ch in chunks
        ]
        for f in as_completed(futs):
            try:
                print(f"[judge] wrote {f.result()} rows")
            except Exception as e:
                print(f"[err] a judge request failed: {e}")


def _run_post_processing(out_dir: Path, judge_model_safe: str) -> None:
    """Write flip-positive and flip-strong subset files from valid_*.jsonl."""
    model_valid = out_dir / f"valid_{judge_model_safe}.jsonl"
    src_valid = model_valid if model_valid.exists() else out_dir / "valid.jsonl"
    if not src_valid.exists():
        print(f"[post] skip flip subsets: {src_valid} not found")
        return
    try:
        dst_flip = out_dir / f"valid_flip_{judge_model_safe}.jsonl"
        kept, total = write_flip_positive(src_valid, dst_flip)
        print(f"[post] flip-positive kept {kept}/{total} -> {dst_flip}")

        dst_flip_strong = out_dir / f"valid_flip_strong_{judge_model_safe}.jsonl"
        kept2, total2 = write_flip_strong(src_valid, dst_flip_strong, min_conf=0.6)
        print(f"[post] flip-strong kept {kept2}/{total2} -> {dst_flip_strong}")
    except Exception as e:
        print(f"[post] flip subsets failed: {e}")


def cmd_run(args):
    if args.only_gen and args.only_valid:
        print("[error] --only-gen and --only-valid cannot both be set.")
        return

    gen_client = get_client(args.gen_model) if not args.only_valid else None
    judge_client = get_client(args.judge_model) if not args.only_gen else None

    run_id = args.run_id or f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    args.run_id = run_id
    out_dir = Path(args.out_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] RUN_ID={run_id}  OUT_DIR={out_dir}")

    prompts = get_prompts_paths(args)
    out_gen = out_dir / "gen.jsonl"
    judge_model_safe = args.judge_model.replace("/", "_").replace(":", "_")
    out_valid = out_dir / f"valid_{judge_model_safe}.jsonl"

    _run_generation(args, gen_client, out_dir, out_gen, prompts)

    if args.only_gen:
        print(f"[done] run complete. Files in {out_dir}")
        return

    print(f"[val] preparing validation with model={args.judge_model}...")
    _run_validation(args, judge_client, out_dir, out_gen, out_valid, prompts)

    _run_post_processing(out_dir, judge_model_safe)
    print(f"[done] run complete. Files in {out_dir}")


# -----------------------------
# CLI
# -----------------------------

def main():
    DEFAULT_TOPICS = [
        "healthcare", "finance", "travel", "technology", "office", "home", "sports", "legal"
    ]
    DEFAULT_MICRO_CUES = [
        "contrast", "quantity", "timing", "location", "role",
        "modality", "negation", "alternative", "priority", "urgency"
    ]

    ap = argparse.ArgumentParser(prog="generate_pairs")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate and validate contrastive sentence pairs")
    g.add_argument("--out-dir", default="runs")
    g.add_argument("--run-id")
    g.add_argument("--prompt-dir", type=Path, default=None,
                   help=f"Dir containing {FN_GEN_SYS}, {FN_GEN_USR}, {FN_JUDGE_SYS}, {FN_JUDGE_USR}")
    g.add_argument("--gen-model", default="gpt-4o-mini")
    g.add_argument("--gen-total", type=int, default=100)
    g.add_argument("--gen-items-per-req", type=int, default=25)
    g.add_argument("--gen-concurrency", type=int, default=1,
                   help="Number of concurrent generation requests")
    g.add_argument("--topics", nargs="*", help="Override default topics")
    g.add_argument("--micro-cues", nargs="*", help="Override default micro cues")
    g.add_argument("--min-len", type=int, default=6)
    g.add_argument("--max-len", type=int, default=12)
    g.add_argument("--judge-model", default="gpt-4o-mini")
    g.add_argument("--judge-chunk-size", type=int, default=40)
    g.add_argument("--judge-concurrency", type=int, default=1,
                   help="Number of concurrent judge requests")
    g.add_argument("--ban-list", nargs="+", default=[],
                   help="JSONL files or HF datasets to load banned words from")
    g.add_argument("--slice-size", type=int, default=0,
                   help="Ban slice size per wave. 0 = use all banned words without slicing")
    g.add_argument("--filter-banned", action="store_true", default=False,
                   help="Drop items whose stressed_word is in the ban list before validation")
    g.add_argument("--validate-pairs", action="store_true", default=False,
                   help="Only validate groups that have >= 2 siblings")
    g.add_argument("--only-gen", action="store_true", default=False,
                   help="Run only the generation stage, skip validation")
    g.add_argument("--only-valid", action="store_true", default=False,
                   help="Run only the validation stage (gen.jsonl must already exist)")
    g.set_defaults(func=cmd_run, DEFAULT_TOPICS=DEFAULT_TOPICS, DEFAULT_MICRO_CUES=DEFAULT_MICRO_CUES)

    args = ap.parse_args()
    if args.cmd == "generate":
        if not args.topics:
            args.topics = args.DEFAULT_TOPICS
        if not args.micro_cues:
            args.micro_cues = args.DEFAULT_MICRO_CUES
    args.func(args)

if __name__ == "__main__":
    main()
