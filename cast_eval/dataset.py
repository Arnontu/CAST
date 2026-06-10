"""Dataset loading utilities for CAST evaluation."""
from pathlib import Path
from datasets import load_dataset, load_from_disk, Dataset


def remove_none_columns_and_report(ds):
    """Remove columns where all values are None, return (ds, removed_cols, kept_cols)."""
    removed_cols = []
    kept_cols = []
    cols_to_remove = []
    for col in ds.column_names:
        if all(v is None for v in ds[col]):
            cols_to_remove.append(col)
            removed_cols.append(col)
        else:
            kept_cols.append(col)
    if cols_to_remove:
        ds = ds.remove_columns(cols_to_remove)
    return ds, removed_cols, kept_cols


def load_test_dataset(dataset_path: str):
    """
    Load a CAST evaluation dataset from a path.

    Accepts:
        - Path to a JSONL file
        - Path to a HuggingFace Arrow dataset directory
    """
    if dataset_path.endswith(".jsonl"):
        ds = load_dataset("json", data_files=dataset_path, split="train")
        ds, _, _ = remove_none_columns_and_report(ds)
    elif Path(dataset_path).exists():
        ds = load_from_disk(dataset_path)
    else:
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    return ds
