"""Evaluate a trained PromptShield checkpoint and write the full evaluation bundle."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
CACHE_DIR = ARTIFACTS_DIR / "cache"
DEFAULT_CHECKPOINT = ARTIFACTS_DIR / "best_checkpoint"


def _require_modules() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import torch
        from datasets import load_from_disk
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Evaluation dependencies are required. Install requirements-train.txt first."
        ) from exc
    return np, torch, (load_from_disk, AutoModelForSequenceClassification, AutoTokenizer)


def _limit(dataset: Any, limit: int | None) -> Any:
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def _load_cached(load_from_disk: Any, name: str, limit: int | None) -> Any:
    path = CACHE_DIR / name
    if not path.is_dir():
        raise FileNotFoundError(
            f"Missing cached dataset {path}. Run `python data.py --limit {limit or 200}` first."
        )
    dataset = _limit(load_from_disk(str(path)), limit)
    labels: dict[str, int] = {}
    for label in dataset["label"]:
        labels[str(label)] = labels.get(str(label), 0) + 1
    print(f"{name}: shape=({len(dataset)}, {len(dataset.column_names)})")
    print(f"{name}: label_distribution={labels}")
    for row in dataset.select(range(min(3, len(dataset)))):
        print(json.dumps(row, ensure_ascii=True, default=str))
    return dataset


def _scores(model: Any, tokenizer: Any, dataset: Any, torch: Any, batch_size: int = 16) -> list[float]:
    model.eval()
    values: list[float] = []
    texts = list(dataset["text"])
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            values.extend(torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist())
    return values


def _choose_threshold(labels: list[int], scores: list[float]) -> tuple[float, dict[str, Any]]:
    """Choose from validation only; this function must never receive test data."""
    if len(labels) != len(scores):
        raise ValueError("Validation labels and scores must have equal lengths.")
    candidates = sorted(set(float(score) for score in scores), reverse=True)
    candidates.append(0.0)
    for threshold in candidates:
        predictions = [int(score >= threshold) for score in scores]
        positives = sum(label == 1 for label in labels)
        true_positives = sum(label == 1 and prediction == 1 for label, prediction in zip(labels, predictions))
        recall = true_positives / positives if positives else 0.0
        if recall >= 0.95:
            return threshold, {
                "source": "primary_val",
                "target_malicious_recall": 0.95,
                "validation_recall": recall,
                "fitted_on_test": False,
            }
    raise RuntimeError("No validation threshold reached malicious recall >= 95%.")


def _metric_values(labels: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    predictions = [int(score >= threshold) for score in scores]
    tn, fp, fn, tp = (int(value) for value in confusion_matrix(labels, predictions, labels=[0, 1]).ravel())
    return {
        "count": len(labels),
        "threshold": threshold,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else None,
        "pr_auc": float(average_precision_score(labels, scores)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "fnr": fn / (fn + tp) if fn + tp else 0.0,
    }


def _plot_outputs(
    results: dict[str, dict[str, Any]],
    labels_by_name: dict[str, list[int]],
    scores_by_name: dict[str, list[float]],
    threshold: float,
    categories_by_name: dict[str, list[str]],
) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, precision_recall_curve

    precision_recall_path = ARTIFACTS_DIR / "pr_curve.png"
    plt.figure()
    for name, labels in labels_by_name.items():
        precision, recall, _ = precision_recall_curve(labels, scores_by_name[name])
        plt.plot(recall, precision, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(precision_recall_path)
    plt.close()

    for name, metric in results.items():
        matrix = metric["confusion_matrix"]
        plt.figure()
        plt.imshow(matrix, cmap="Blues")
        plt.colorbar()
        plt.xticks([0, 1], ["benign", "malicious"])
        plt.yticks([0, 1], ["benign", "malicious"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion matrix: {name}")
        for row in range(2):
            for column in range(2):
                plt.text(column, row, matrix[row][column], ha="center", va="center")
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / f"confusion_matrix_{name}.png")
        plt.close()

    category_rows = _category_rows(
        labels_by_name, scores_by_name, threshold, categories_by_name
    )
    categories = [f"{row['dataset']}:{row['category']}" for row in category_rows]
    recalls = [float(row["recall"]) if row["recall"] is not None else 0.0 for row in category_rows]
    plt.figure(figsize=(max(8, len(categories) * 0.35), 5))
    plt.bar(categories, recalls)
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Recall")
    plt.title("Per-category recall")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "per_category_recall.png")
    plt.close()


def _category_rows(
    labels_by_name: dict[str, list[int]],
    scores_by_name: dict[str, list[float]],
    threshold: float,
    categories_by_name: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    categories_by_name = categories_by_name or {}
    for name, labels in labels_by_name.items():
        categories = categories_by_name.get(name, ["unknown"] * len(labels))
        grouped: dict[str, list[int]] = {}
        for index, category in enumerate(categories):
            grouped.setdefault(str(category) or "unknown", []).append(index)
        for category, indices in sorted(grouped.items()):
            positives = [labels[index] for index in indices]
            predictions = [
                int(scores_by_name[name][index] >= threshold) for index in indices
            ]
            actual_positive = sum(positives)
            true_positive = sum(
                actual == 1 and predicted == 1
                for actual, predicted in zip(positives, predictions)
            )
            rows.append(
                {
                    "dataset": name,
                    "category": category,
                    "count": len(indices),
                    "malicious_count": actual_positive,
                    "recall": true_positive / actual_positive if actual_positive else 0.0,
                    "precision": (
                        true_positive / sum(predictions) if sum(predictions) else 0.0
                    ),
                    "f1": (
                        2 * true_positive / (actual_positive + sum(predictions))
                        if actual_positive + sum(predictions)
                        else 0.0
                    ),
                }
            )
    return rows


def _latency(model: Any, tokenizer: Any, torch: Any, text: str) -> dict[str, Any]:
    torch.set_num_threads(1)
    def run() -> None:
        encoded = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            model(**encoded)
    for _ in range(20):
        run()
    samples: list[float] = []
    for _ in range(200):
        start = time.perf_counter()
        run()
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    return {
        "device": "cpu",
        "threads": 1,
        "batch_size": 1,
        "warmup_runs": 20,
        "measured_runs": 200,
        "p50_ms": samples[99],
        "p95_ms": samples[189],
    }


def _write_predictions(
    rows: list[dict[str, Any]],
) -> None:
    fields = ["dataset", "index", "text", "category", "label", "score", "prediction", "correct"]
    with (ARTIFACTS_DIR / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(checkpoint: str | Path = DEFAULT_CHECKPOINT, limit: int | None = None) -> None:
    np, torch, modules = _require_modules()
    load_from_disk, AutoModelForSequenceClassification, AutoTokenizer = modules
    checkpoint_ref = str(checkpoint)
    is_hub_id = (
        "/" in checkpoint_ref
        and not checkpoint_ref.startswith("./")
        and not checkpoint_ref.startswith("/")
    )
    if not is_hub_id:
        checkpoint_path = Path(checkpoint_ref)
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(
                f"Checkpoint directory does not exist: {checkpoint_path}"
            )
    datasets = {
        "primary_val": _load_cached(load_from_disk, "primary_val", limit),
        "primary_test": _load_cached(load_from_disk, "primary_test", limit),
        "cross_source_test": _load_cached(load_from_disk, "cross_source_test_audit_filtered", limit)
        if (CACHE_DIR / "cross_source_test_audit_filtered").is_dir()
        else _load_cached(load_from_disk, "cross_source_test", limit),
    }
    wild_path = CACHE_DIR / "wildjailbreak_eval_audit_filtered"
    if not wild_path.is_dir():
        wild_path = CACHE_DIR / "wildjailbreak_eval"
    if wild_path.is_dir():
        wild = _load_cached(load_from_disk, wild_path.name, limit)
        wild = wild.filter(lambda row: row["data_type"] == "adversarial_benign")
        datasets["wildjailbreak_adversarial_benign"] = wild
    else:
        print("WARNING: WildJailbreak cache unavailable; skipping adversarial_benign evaluation.")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_ref)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_ref)
    model.to("cpu")
    validation_scores = _scores(model, tokenizer, datasets["primary_val"], torch)
    threshold, threshold_info = _choose_threshold(
        list(datasets["primary_val"]["label"]),
        validation_scores,
    )
    assert threshold_info["fitted_on_test"] is False

    results: dict[str, dict[str, Any]] = {}
    labels_by_name: dict[str, list[int]] = {}
    scores_by_name: dict[str, list[float]] = {}
    categories_by_name: dict[str, list[str]] = {}
    prediction_rows: list[dict[str, Any]] = []
    for name, dataset in datasets.items():
        if name == "primary_val":
            continue
        labels = list(dataset["label"])
        scores = _scores(model, tokenizer, dataset, torch)
        results[name] = _metric_values(labels, scores, threshold)
        labels_by_name[name] = labels
        scores_by_name[name] = scores
        categories_by_name[name] = list(dataset["category"])
        for index, (text, category, label, score) in enumerate(
            zip(dataset["text"], dataset["category"], labels, scores)
        ):
            prediction = int(score >= threshold)
            prediction_rows.append(
                {
                    "dataset": name,
                    "index": index,
                    "text": text,
                    "category": category,
                    "label": label,
                    "score": score,
                    "prediction": prediction,
                    "correct": prediction == label,
                }
            )

    category_rows = _category_rows(
        labels_by_name, scores_by_name, threshold, categories_by_name
    )
    errors = [row for row in prediction_rows if not row["correct"]]
    errors.sort(
        key=lambda row: row["score"] if row["prediction"] == 1 else 1 - row["score"],
        reverse=True,
    )
    error_fields = [
        "dataset", "index", "text", "category", "label", "score", "prediction", "confidence"
    ]
    with (ARTIFACTS_DIR / "errors.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=error_fields)
        writer.writeheader()
        for row in errors[:40]:
            row = dict(row)
            row["confidence"] = row["score"] if row["prediction"] else 1 - row["score"]
            writer.writerow({field: row[field] for field in error_fields})

    latency = _latency(model, tokenizer, torch, datasets["primary_test"]["text"][0])
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    metrics["main_model"] = results
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (ARTIFACTS_DIR / "threshold.json").write_text(
        json.dumps({"main_model": {"threshold": threshold, **threshold_info}}, indent=2) + "\n",
        encoding="utf-8",
    )
    (ARTIFACTS_DIR / "latency.json").write_text(json.dumps(latency, indent=2) + "\n", encoding="utf-8")
    with (ARTIFACTS_DIR / "per_category.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "category", "count", "malicious_count", "recall"])
        writer.writeheader()
        writer.writerows(category_rows)
    _write_predictions(prediction_rows)
    _plot_outputs(
        results,
        labels_by_name,
        scores_by_name,
        threshold,
        categories_by_name,
    )
    with (ARTIFACTS_DIR / "results_table.md").open("w", encoding="utf-8") as handle:
        handle.write("| Dataset | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR | FNR |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for name, metric in results.items():
            handle.write(
                f"| {name} | {metric['precision']:.4f} | {metric['recall']:.4f} | "
                f"{metric['f1']:.4f} | {metric['roc_auc'] if metric['roc_auc'] is not None else 'TODO'} | "
                f"{metric['pr_auc']:.4f} | {metric['fpr']:.4f} | {metric['fnr']:.4f} |\n"
            )
    print(f"Frozen validation threshold: {threshold}")
    print(f"Wrote evaluation artifacts to {ARTIFACTS_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    return args


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print("Dry run: no checkpoint, datasets, or plots loaded.")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Row limit per dataset: {args.limit if args.limit is not None else 'unlimited'}")
        return
    evaluate(args.checkpoint, args.limit)


if __name__ == "__main__":
    main()
