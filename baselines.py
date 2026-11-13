"""Train and evaluate the PromptShield CPU baselines."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
CACHE_DIR = ARTIFACTS_DIR / "cache"
PRIMARY_ID = "neuralchemy/Prompt-injection-dataset"
ZERO_SHOT_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"


def _require_datasets() -> Any:
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install requirements-train.txt first."
        ) from exc
    return load_dataset, load_from_disk


def _require_sklearn() -> Any:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            average_precision_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is required. Install requirements-train.txt first."
        ) from exc
    return (
        TfidfVectorizer,
        LogisticRegression,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        Pipeline,
    )


def _limit(dataset: Any, limit: int | None) -> Any:
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def _print_dataset(dataset: Any, name: str) -> None:
    labels: dict[str, int] = {}
    for label in dataset["label"]:
        labels[str(label)] = labels.get(str(label), 0) + 1
    print(f"{name}: shape=({len(dataset)}, {len(dataset.column_names)})")
    print(f"{name}: label_distribution={labels}")
    print(f"{name}: sample_rows=")
    for row in dataset.select(range(min(3, len(dataset)))):
        print(json.dumps(row, ensure_ascii=True, default=str))


def _load_core(load_dataset: Any, limit: int | None) -> dict[str, Any]:
    dataset_dict = load_dataset(
        PRIMARY_ID,
        "core",
        cache_dir=str(CACHE_DIR / "downloads"),
    )
    keys = list(dataset_dict.keys())
    print(f"Primary core actual split keys: {keys}")
    mapping = {"train": "train", "validation": "val", "test": "test"}
    missing = set(mapping) - set(keys)
    if missing:
        raise RuntimeError(
            f"Primary core is missing required split(s) {sorted(missing)}; "
            f"available splits: {keys}"
        )
    result = {internal: _limit(dataset_dict[real], limit) for real, internal in mapping.items()}
    for split, dataset in result.items():
        _print_dataset(dataset, f"core/{split}")
    return result


def _load_primary_test(load_from_disk: Any, load_dataset: Any, limit: int | None) -> Any:
    cache_path = CACHE_DIR / "primary_test"
    if cache_path.is_dir():
        dataset = _limit(load_from_disk(str(cache_path)), limit)
        _print_dataset(dataset, "primary/test")
        return dataset
    dataset_dict = load_dataset(
        PRIMARY_ID,
        "full",
        cache_dir=str(CACHE_DIR / "downloads"),
    )
    keys = list(dataset_dict.keys())
    print(f"Primary full actual split keys: {keys}")
    if "test" not in keys:
        raise RuntimeError(
            f"Primary full is missing required split 'test'; available splits: {keys}"
        )
    dataset = _limit(dataset_dict["test"], limit)
    _print_dataset(dataset, "primary/test")
    return dataset


def _binary_guard(dataset: Any, name: str) -> None:
    labels = set(dataset["label"])
    if labels != {0, 1}:
        raise RuntimeError(f"{name} must contain both labels 0 and 1; found {sorted(labels)}")


def _choose_threshold(y_true: list[int], scores: list[float], target_recall: float = 0.95) -> tuple[float, dict[str, Any]]:
    candidates = sorted(set(scores), reverse=True)
    candidates.append(0.0)
    selected: tuple[float, dict[str, Any]] | None = None
    for threshold in candidates:
        predictions = [int(score >= threshold) for score in scores]
        tp = sum(actual == 1 and predicted == 1 for actual, predicted in zip(y_true, predictions))
        fn = sum(actual == 1 and predicted == 0 for actual, predicted in zip(y_true, predictions))
        recall = tp / (tp + fn) if tp + fn else 0.0
        if recall >= target_recall:
            selected = (threshold, {"target_recall": target_recall, "recall": recall})
            break
    if selected is None:
        raise RuntimeError("Could not select a threshold meeting the required malicious recall.")
    return selected


def _metrics(y_true: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
    (
        _,
        _,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        _,
    ) = _require_sklearn()
    predictions = [int(score >= threshold) for score in scores]
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    return {
        "count": len(y_true),
        "threshold": threshold,
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "fpr": fp / (fp + tn) if fp + tn else math.nan,
        "fnr": fn / (fn + tp) if fn + tp else math.nan,
    }


def _run_tfidf(core: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    (
        TfidfVectorizer,
        LogisticRegression,
        _,
        _,
        _,
        _,
        _,
        _,
        Pipeline,
    ) = _require_sklearn()
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(core["train"]["text"], core["train"]["label"])
    val_scores = model.predict_proba(core["val"]["text"])[:, 1].tolist()
    threshold, threshold_info = _choose_threshold(
        list(core["val"]["label"]),
        val_scores,
    )
    test_scores = model.predict_proba(core["test"]["text"])[:, 1].tolist()
    return (
        _metrics(list(core["test"]["label"]), test_scores, threshold),
        {
            "baseline_tfidf": {
                "threshold": threshold,
                "validation": threshold_info,
                "selection": "highest score threshold meeting malicious recall >= 0.95",
            }
        },
    )


def _malicious_label_index(model: Any) -> int:
    labels = {int(index): str(label).lower() for index, label in model.config.id2label.items()}
    for index, label in labels.items():
        if index == 1 or any(term in label for term in ("malicious", "injection", "jailbreak")):
            return index
    raise RuntimeError(f"Could not identify malicious model label from id2label={labels}")


def _run_zero_shot(test: Any, limit: int | None) -> dict[str, Any]:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for Baseline B. Install requirements-train.txt first."
        ) from exc
    classifier = pipeline(
        "text-classification",
        model=ZERO_SHOT_MODEL,
        tokenizer=ZERO_SHOT_MODEL,
        device=-1,
        top_k=None,
    )
    outputs = classifier(
        list(test["text"]),
        truncation=True,
        max_length=512,
        batch_size=8,
    )

    if isinstance(outputs, dict):
        outputs = [[outputs]]
    elif outputs and isinstance(outputs[0], dict):
        outputs = [outputs]

    scores: list[float] = []
    for output in outputs:
        malicious_score: float | None = None
        for item in output:
            label = str(item["label"]).strip().upper()
            if label in {"INJECTION", "PROMPT_INJECTION", "LABEL_1"}:
                malicious_score = float(item["score"])
                break
            if label in {"SAFE", "LABEL_0"}:
                continue
        if malicious_score is None:
            labels = [str(item.get("label", "")) for item in output]
            raise RuntimeError(
                "Could not identify the malicious class in zero-shot output; "
                f"received labels: {labels}"
            )
        scores.append(malicious_score)
    return _metrics(list(test["label"]), scores, 0.5)


def _merge_json(path: Path, values: dict[str, Any]) -> None:
    existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    existing.update(values)
    path.write_text(json.dumps(existing, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows per loaded split.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned runs without loading models.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    return args


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print("Dry run: no datasets loaded and no models downloaded.")
        print(f"Baseline A: TF-IDF + LogisticRegression on {PRIMARY_ID} config=core")
        print(f"Baseline B: {ZERO_SHOT_MODEL} on primary full test")
        print(f"Row limit per split: {args.limit if args.limit is not None else 'unlimited'}")
        return
    load_dataset, load_from_disk = _require_datasets()
    core = _load_core(load_dataset, args.limit)
    for dataset in core.values():
        _binary_guard(dataset, "Core split")
    test = _load_primary_test(load_from_disk, load_dataset, args.limit)
    _binary_guard(test, "Primary test")
    tfidf_metrics, thresholds = _run_tfidf(core)
    zero_shot_metrics = _run_zero_shot(test, args.limit)
    _merge_json(ARTIFACTS_DIR / "metrics.json", {
        "baseline_tfidf": tfidf_metrics,
        "baseline_deberta_zero_shot": zero_shot_metrics,
    })
    thresholds["baseline_deberta_zero_shot"] = {
        "threshold": 0.5,
        "selection": "model score threshold; no validation tuning",
    }
    _merge_json(ARTIFACTS_DIR / "threshold.json", thresholds)
    print(json.dumps({
        "baseline_tfidf": tfidf_metrics,
        "baseline_deberta_zero_shot": zero_shot_metrics,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
