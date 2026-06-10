#!/usr/bin/env python3
"""Build a CAST eval dataset (HuggingFace Arrow) from one or more valid_*.jsonl files."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Value, Sequence


def load_valid_jsonls(paths: list[Path]) -> list[dict[str, Any]]:
    """Read all items from one or more valid_*.jsonl files."""
    items = []
    for path in paths:
        if not path.exists():
            print(f"[warn] not found: {path}", file=sys.stderr)
            continue
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[warn] {path}:{lineno}: {e}", file=sys.stderr)
    return items


def filter_judge_match(items: list[dict]) -> list[dict]:
    """Keep only items where the judge agreed with the generator's stressed word."""
    return [it for it in items if it.get("judge_match") is True]


def dedup_by_item_id(items: list[dict]) -> list[dict]:
    """Deduplicate by item_id; first occurrence wins (preserves input file order)."""
    seen: set[str] = set()
    out = []
    for it in items:
        iid = it.get("item_id")
        if iid and iid in seen:
            continue
        if iid:
            seen.add(iid)
        out.append(it)
    return out


def filter_complete_pairs(items: list[dict], min_pairs: int = 1) -> list[dict]:
    """Keep only items whose group_id has at least 2 siblings present."""
    group_counts = Counter(it.get("group_id") for it in items if it.get("group_id"))
    valid_groups = {gid for gid, cnt in group_counts.items() if cnt >= max(2, min_pairs)}
    return [it for it in items if it.get("group_id") in valid_groups]


def add_file_name(items: list[dict]) -> list[dict]:
    """Add file_name = item_id + '.wav' so run_eval.py can locate synthesized audio."""
    out = []
    for it in items:
        row = dict(it)
        if "file_name" not in row:
            row["file_name"] = (row.get("item_id") or "") + ".wav"
        out.append(row)
    return out


_FEATURES = Features({
    "sentence": Value("string"),
    "context": Value("string"),
    "stressed_word": Value("string"),
    "judge_match": Value("bool"),
    "mask": Sequence(Value("int64")),
    "stressed_index": Value("int64"),
    "item_id": Value("string"),
    "group_id": Value("string"),
    "gen_model": Value("string"),
    "valid_model": Value("string"),
    "pos_code": Value("string"),
    "mode_code": Value("string"),
    "ctx_code": Value("string"),
    "gen_style": Value("string"),
    "judge_word": Value("string"),
    "judge_confidence": Value("float64"),
    "judge_cue": Value("string"),
    "judge_trigger": Value("string"),
    "judge_flip": Value("bool"),
    "judge_cc_quality_ok": Value("bool"),
    "run_id": Value("string"),
    "step": Value("string"),
    "file_name": Value("string"),
})


def build_and_save(items: list[dict], out_dir: Path) -> Dataset:
    """Create a HuggingFace Dataset from items and save to disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Coerce items to match the declared schema (fill missing fields with None)
    field_names = set(_FEATURES.keys())
    rows = [{k: row.get(k) for k in field_names} for row in items]
    ds = Dataset.from_list(rows, features=_FEATURES)
    ds.save_to_disk(str(out_dir))
    return ds


def resolve_inputs(raw_inputs: list[str]) -> list[Path]:
    """Accept paths to .jsonl files or run dirs (globs valid_*.jsonl inside)."""
    paths = []
    for s in raw_inputs:
        p = Path(s)
        if p.is_file():
            paths.append(p)
        elif p.is_dir():
            found = sorted(p.glob("valid_*.jsonl"))
            if not found:
                print(f"[warn] no valid_*.jsonl in {p}", file=sys.stderr)
            paths.extend(found)
        else:
            print(f"[warn] not found: {p}", file=sys.stderr)
    return paths


def main():
    ap = argparse.ArgumentParser(description="Build a CAST eval dataset from validated JSONL files")
    ap.add_argument("inputs", nargs="+",
                    help="Paths to valid_*.jsonl files or run directories")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Output directory for the HuggingFace Arrow dataset")
    ap.add_argument("--min-pairs", type=int, default=1,
                    help="Minimum siblings per group_id (default: 1, requires exactly 2)")
    ap.add_argument("--skip-judge-filter", action="store_true",
                    help="Skip the judge_match=True filter")
    args = ap.parse_args()

    paths = resolve_inputs(args.inputs)
    if not paths:
        print("[error] no valid input files found", file=sys.stderr)
        sys.exit(1)

    print(f"[info] loading from {len(paths)} file(s)...")
    items = load_valid_jsonls(paths)
    print(f"[info] loaded {len(items)} items")

    if not args.skip_judge_filter:
        items = filter_judge_match(items)
        print(f"[info] after judge_match filter: {len(items)}")

    items = dedup_by_item_id(items)
    print(f"[info] after dedup: {len(items)}")

    before_pairs = len(items)
    items = filter_complete_pairs(items, min_pairs=args.min_pairs)
    n_pairs = len({it["group_id"] for it in items})
    print(f"[info] after pair filter: {len(items)} items in {n_pairs} pairs "
          f"(dropped {before_pairs - len(items)} orphans)")

    items = add_file_name(items)

    if not items:
        print("[error] no items remaining after filters", file=sys.stderr)
        sys.exit(1)

    ds = build_and_save(items, args.out_dir)
    print(f"[info] saved {len(ds)} items → {args.out_dir}")

    summary = {
        "n_items": len(items),
        "n_pairs": n_pairs,
        "source_files": [str(p) for p in paths],
        "filters": {
            "judge_match": not args.skip_judge_filter,
            "min_pairs": args.min_pairs,
        },
    }
    (args.out_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[done] dataset ready at {args.out_dir}")
    print(f"       load with: load_test_dataset('{args.out_dir}')")


if __name__ == "__main__":
    main()
