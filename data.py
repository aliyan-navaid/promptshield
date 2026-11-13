"""Download, normalize, inspect, and cache PromptShield datasets.

The WildJailbreak dataset is intentionally kept separate from training data:
its labels describe harmfulness, not prompt injection.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "artifacts" / "cache"
PRIMARY_ID = "neuralchemy/Prompt-injection-dataset"
CROSS_SOURCE_ID = "S-Labs/prompt-injection-dataset"
WILDJAILBREAK_ID = "allenai/wildjailbreak"
COMMON_COLUMNS = [
    "text",
    "label",
    "category",
    "severity",
    "group_id",
    "augmented",
    "source",
    "tags",
    "dataset",
    "split",
    "evaluation_only",
    "data_type",
]
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\ufeff]")


def _require_datasets() -> Any:
    try:
        from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install requirements-train.txt first."
        ) from exc
    return Dataset, DatasetDict, load_dataset, load_from_disk


def _limit_dataset(dataset: Any, limit: int | None) -> Any:
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def _value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key, default)
    return default if value is None else value


def _normalize(
    dataset: Any,
    *,
    dataset_name: str,
    split: str,
    limit: int | None,
    evaluation_only: bool,
    wildjailbreak: bool = False,
) -> Any:
    Dataset, _, _, _ = _require_datasets()
    dataset = _limit_dataset(dataset, limit)
    rows: list[dict[str, Any]] = []
    for row in dataset:
        if wildjailbreak:
            data_type = str(_value(row, "data_type", ""))
            label = 1 if data_type == "adversarial_harmful" else 0
            text = _value(row, "adversarial", "")
            category = data_type
        else:
            label = int(_value(row, "label", 0))
            text = _value(row, "text", "")
            category = _value(row, "category", "")
        rows.append(
            {
                "text": str(text),
                "label": label,
                "category": category,
                "severity": _value(row, "severity", ""),
                "group_id": _value(row, "group_id", ""),
                "augmented": _value(row, "augmented", False),
                "source": _value(row, "source", dataset_name),
                "tags": _value(row, "tags", []),
                "dataset": dataset_name,
                "split": split,
                "evaluation_only": evaluation_only,
                "data_type": _value(row, "data_type", ""),
            }
        )
    return Dataset.from_list(rows).select_columns(COMMON_COLUMNS)


def _print_dataset(dataset: Any, name: str) -> None:
    labels: dict[str, int] = {}
    for label in dataset["label"]:
        key = str(label)
        labels[key] = labels.get(key, 0) + 1
    print(f"\n{name}: shape=({len(dataset)}, {len(dataset.column_names)})")
    print(f"{name}: label_distribution={labels}")
    print(f"{name}: sample_rows=")
    for row in dataset.select(range(min(3, len(dataset)))):
        print(json.dumps(row, ensure_ascii=True, default=str))


def _cache_split(dataset: Any, cache_path: Path) -> None:
    if cache_path.exists():
        shutil.rmtree(cache_path)
    dataset.save_to_disk(str(cache_path))


def _split_keys(dataset_dict: Any, dataset_name: str) -> list[str]:
    keys = list(dataset_dict.keys())
    print(f"{dataset_name} actual split keys: {keys}")
    return keys


def _require_splits(
    dataset_dict: Any,
    *,
    dataset_name: str,
    mapping: dict[str, str],
) -> dict[str, Any]:
    keys = _split_keys(dataset_dict, dataset_name)
    missing = {real_name for real_name in mapping if real_name not in keys}
    if missing:
        raise RuntimeError(
            f"{dataset_name} is missing required split(s) {sorted(missing)}; "
            f"available splits: {keys}"
        )
    return {
        internal_name: dataset_dict[real_name]
        for real_name, internal_name in mapping.items()
    }


def _load_primary(load_dataset: Any, limit: int | None) -> dict[str, Any]:
    dataset_dict = load_dataset(
        PRIMARY_ID,
        "full",
        cache_dir=str(CACHE_DIR / "downloads"),
    )
    splits = _require_splits(
        dataset_dict,
        dataset_name="Primary",
        mapping={"train": "train", "validation": "val", "test": "test"},
    )
    return {
        split_name: _normalize(
            dataset,
            dataset_name="primary",
            split=split_name,
            limit=limit,
            evaluation_only=False,
        )
        for split_name, dataset in splits.items()
    }


def _load_cross_source(load_dataset: Any, limit: int | None) -> Any:
    dataset_dict = load_dataset(
        CROSS_SOURCE_ID,
        cache_dir=str(CACHE_DIR / "downloads"),
    )
    dataset = _require_splits(
        dataset_dict,
        dataset_name="Cross-source",
        mapping={"test": "test"},
    )["test"]
    return _normalize(
        dataset,
        dataset_name="cross_source",
        split="test",
        limit=limit,
        evaluation_only=True,
    )


def _load_wildjailbreak(load_dataset: Any, limit: int | None) -> Any:
    from huggingface_hub.errors import GatedRepoError

    try:
        dataset_dict = load_dataset(
            WILDJAILBREAK_ID,
            "eval",
            delimiter="\t",
            keep_default_na=False,
            cache_dir=str(CACHE_DIR / "downloads"),
        )
    except GatedRepoError:
        print(
            "ACTION REQUIRED: Go to "
            "https://huggingface.co/datasets/allenai/wildjailbreak, "
            "log in, and click 'Agree and access repository'. Then rerun."
        )
        return None
    keys = _split_keys(dataset_dict, "WildJailbreak")
    real_split = "train" if "train" in keys else next(
        (key for key in ("eval", "test") if key in keys),
        None,
    )
    if real_split is None:
        raise RuntimeError(
            "WildJailbreak has no usable evaluation split; "
            f"available splits: {keys}"
        )
    dataset = dataset_dict[real_split]
    if "data_type" not in dataset.column_names:
        raise RuntimeError(
            "WildJailbreak eval split is missing required column 'data_type'; "
            f"available columns: {dataset.column_names}"
        )
    return _normalize(
        dataset,
        dataset_name="wildjailbreak",
        split="test",
        limit=limit,
        evaluation_only=True,
        wildjailbreak=True,
    )


def load_all(limit: int | None = None) -> dict[str, Any]:
    """Load all requested data, normalize it, print diagnostics, and cache it."""
    _, _, load_dataset, _ = _require_datasets()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, Any] = {}

    primary = _load_primary(load_dataset, limit)
    for split, dataset in primary.items():
        datasets[f"primary_{split}"] = dataset
        _print_dataset(dataset, f"primary/{split}")
        _cache_split(dataset, CACHE_DIR / f"primary_{split}")

    cross_source = _load_cross_source(load_dataset, limit)
    datasets["cross_source_test"] = cross_source
    _print_dataset(cross_source, "cross_source/test")
    _cache_split(cross_source, CACHE_DIR / "cross_source_test")

    wild_status = {"available": False, "warning": None}
    try:
        wild = _load_wildjailbreak(load_dataset, limit)
    except Exception as exc:  # gated datasets can fail with several hub exceptions
        wild_status["warning"] = f"WildJailbreak unavailable; skipping it: {exc}"
        print(f"WARNING: {wild_status['warning']}", file=sys.stderr)
    else:
        if wild is not None:
            wild_status["available"] = True
            datasets["wildjailbreak_eval"] = wild
            _print_dataset(wild, "wildjailbreak/eval")
            _cache_split(wild, CACHE_DIR / "wildjailbreak_eval")
    (CACHE_DIR / "wildjailbreak_status.json").write_text(
        json.dumps(wild_status, indent=2) + "\n", encoding="utf-8"
    )
    return datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows per loaded split.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned dataset loads without downloading data.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    return args


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print("Dry run: no datasets downloaded or cached.")
        print(f"Primary: {PRIMARY_ID}, config=full, splits=train/val/test")
        print(f"Cross-source: {CROSS_SOURCE_ID}, split=test")
        print(f"WildJailbreak: {WILDJAILBREAK_ID}, config=eval, split=eval (optional)")
        print(f"Row limit per split: {args.limit if args.limit is not None else 'unlimited'}")
        return
    load_all(args.limit)


if __name__ == "__main__":
    main()
