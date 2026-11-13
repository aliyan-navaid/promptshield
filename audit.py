"""Audit cached PromptShield splits before training.

Run data.py first so the normalized datasets exist under artifacts/cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
CACHE_DIR = ARTIFACTS_DIR / "cache"
PRIMARY_SPLITS = {
    "train": "primary_train",
    "val": "primary_val",
    "test": "primary_test",
}
EXTERNAL_SPLITS = {
    "cross_source_test": "cross_source_test",
    "wildjailbreak_eval": "wildjailbreak_eval",
}
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\ufeff]")


def _require_datasets() -> Any:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install requirements-train.txt first."
        ) from exc
    return load_from_disk


def normalize_text(text: str) -> str:
    """Lowercase, remove zero-width characters, and strip surrounding whitespace."""
    return ZERO_WIDTH_RE.sub("", str(text)).lower().strip()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_cached(load_from_disk: Any, cache_name: str, limit: int | None) -> Any:
    path = CACHE_DIR / cache_name
    if not path.is_dir():
        raise FileNotFoundError(
            f"Missing cached dataset {path}. Run `python data.py` before audit.py."
        )
    dataset = load_from_disk(str(path))
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def _group_integrity(primary: dict[str, Any]) -> dict[str, Any]:
    groups: dict[str, set[str]] = defaultdict(set)
    missing_group_id = 0
    rows_by_split: dict[str, int] = {}
    for split, dataset in primary.items():
        rows_by_split[split] = len(dataset)
        for group_id in dataset["group_id"]:
            value = "" if group_id is None else str(group_id).strip()
            if not value:
                missing_group_id += 1
                continue
            groups[value].add(split)
    spanning = {
        group_id: sorted(splits)
        for group_id, splits in groups.items()
        if len(splits) > 1
    }
    return {
        "passed": not spanning,
        "rows_by_split": rows_by_split,
        "unique_group_ids": len(groups),
        "rows_missing_group_id": missing_group_id,
        "groups_spanning_splits": spanning,
        "groups_spanning_split_count": len(spanning),
    }


def _text_sets(dataset: Any) -> tuple[set[str], dict[str, set[int]]]:
    exact_to_indices: dict[str, set[int]] = defaultdict(set)
    for index, text in enumerate(dataset["text"]):
        exact_to_indices[str(text)].add(index)
    return set(exact_to_indices), exact_to_indices


def _normalized_hashes(dataset: Any) -> tuple[dict[str, set[int]], dict[str, str]]:
    hash_to_indices: dict[str, set[int]] = defaultdict(set)
    normalized_by_hash: dict[str, str] = {}
    for index, text in enumerate(dataset["text"]):
        normalized = normalize_text(str(text))
        digest = _sha256(normalized)
        hash_to_indices[digest].add(index)
        normalized_by_hash[digest] = normalized
    return hash_to_indices, normalized_by_hash


def _overlap(
    primary_train: Any,
    external: Any,
    external_name: str,
) -> tuple[dict[str, Any], set[int]]:
    primary_exact, _ = _text_sets(primary_train)
    external_exact, external_exact_indices = _text_sets(external)
    exact_values = primary_exact & external_exact
    exact_indices = {
        index
        for value in exact_values
        for index in external_exact_indices[value]
    }

    primary_hashes, _ = _normalized_hashes(primary_train)
    external_hashes, _ = _normalized_hashes(external)
    normalized_values = set(primary_hashes) & set(external_hashes)
    normalized_indices = {
        index
        for digest in normalized_values
        for index in external_hashes[digest]
    }
    removed_indices = exact_indices | normalized_indices
    result = {
        "dataset": external_name,
        "primary_train_rows": len(primary_train),
        "external_rows": len(external),
        "exact_overlap_values": len(exact_values),
        "exact_overlap_rows": len(exact_indices),
        "normalized_hash_overlap_values": len(normalized_values),
        "normalized_hash_overlap_rows": len(normalized_indices),
        "removed_indices": sorted(removed_indices),
        "removed_count": len(removed_indices),
        "remaining_count": len(external) - len(removed_indices),
    }
    return result, removed_indices


def _save_filtered(dataset: Any, removed_indices: set[int], cache_name: str) -> None:
    filtered_indices = [
        index for index in range(len(dataset)) if index not in removed_indices
    ]
    output = CACHE_DIR / f"{cache_name}_audit_filtered"
    if output.exists():
        import shutil

        shutil.rmtree(output)
    dataset.select(filtered_indices).save_to_disk(str(output))


def _print_summary(
    group_result: dict[str, Any],
    overlaps: list[dict[str, Any]],
) -> None:
    print("\nAudit summary")
    print("| Check | Dataset | Exact rows | Normalized-hash rows | Removed | Status |")
    print("|---|---|---:|---:|---:|---|")
    print(
        "| Group integrity | primary splits | - | - | - | "
        f"{'PASS' if group_result['passed'] else 'FAIL'} |"
    )
    for result in overlaps:
        print(
            f"| Text overlap | {result['dataset']} | "
            f"{result['exact_overlap_rows']} | "
            f"{result['normalized_hash_overlap_rows']} | "
            f"{result['removed_count']} | "
            f"{'PASS' if result['removed_count'] == 0 else 'REMOVED'} |"
        )


def _print_zero_overlap_verification(
    primary_train: Any,
    external_datasets: dict[str, Any],
    overlaps: list[dict[str, Any]],
) -> None:
    if any(result["removed_count"] for result in overlaps):
        return
    print("\nNo overlap found after exact and normalized-hash comparison.")
    print(
        "Normalization function: "
        "sha256(lowercase(remove_zero_width(text)).strip().encode('utf-8'))"
    )
    hashes: list[str] = []
    for dataset in [primary_train, *external_datasets.values()]:
        for text in dataset["text"]:
            digest = _sha256(normalize_text(str(text)))
            if digest not in hashes:
                hashes.append(digest)
            if len(hashes) == 3:
                break
        if len(hashes) == 3:
            break
    for index, digest in enumerate(hashes, start=1):
        print(f"Example normalized hash {index}: {digest}")


def audit(limit: int | None = None) -> dict[str, Any]:
    load_from_disk = _require_datasets()
    primary = {
        split: _load_cached(load_from_disk, cache_name, limit)
        for split, cache_name in PRIMARY_SPLITS.items()
    }
    external: dict[str, Any] = {}
    for name, cache_name in EXTERNAL_SPLITS.items():
        path = CACHE_DIR / cache_name
        if path.is_dir():
            external[name] = _load_cached(load_from_disk, cache_name, limit)
        elif name == "wildjailbreak_eval":
            print("WARNING: WildJailbreak cache is unavailable; skipping it.")
        else:
            raise FileNotFoundError(
                f"Missing cached external evaluation dataset {path}. "
                "Run `python data.py` before audit.py."
            )

    group_result = _group_integrity(primary)
    overlaps: list[dict[str, Any]] = []
    for name, dataset in external.items():
        result, removed_indices = _overlap(
            primary["train"],
            dataset,
            name,
        )
        overlaps.append(result)
        if removed_indices:
            _save_filtered(dataset, removed_indices, name)
        else:
            _save_filtered(dataset, set(), name)

    audit_result = {
        "normalization": {
            "operation": "lowercase, remove zero-width characters, strip surrounding whitespace",
            "hash": "sha256",
        },
        "limit": limit,
        "group_integrity": group_result,
        "external_overlaps": overlaps,
        "filtered_cache_suffix": "_audit_filtered",
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "audit.json").write_text(
        json.dumps(audit_result, indent=2) + "\n",
        encoding="utf-8",
    )
    _print_summary(group_result, overlaps)
    _print_zero_overlap_verification(primary["train"], external, overlaps)
    return audit_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows per cached split.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned audit without loading or writing datasets.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    return args


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print("Dry run: no datasets loaded, filtered, or written.")
        print(f"Primary cached splits: {', '.join(PRIMARY_SPLITS)}")
        print(f"External cached evaluations: {', '.join(EXTERNAL_SPLITS)}")
        print(f"Row limit per split: {args.limit if args.limit is not None else 'unlimited'}")
        return
    audit(args.limit)


if __name__ == "__main__":
    main()
