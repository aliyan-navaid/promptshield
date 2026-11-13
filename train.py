"""Fine-tune DeBERTa-v3-small on the cached primary PromptShield data."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
CACHE_DIR = ARTIFACTS_DIR / "cache"
CONFIG_PATH = ROOT / "configs" / "deberta_small.yaml"


def _read_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else CONFIG_PATH
    try:
        import yaml
    except ImportError:
        values: dict[str, Any] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, raw = (part.strip() for part in line.split(":", 1))
            if raw.lower() in {"true", "false"}:
                value: Any = raw.lower() == "true"
            else:
                try:
                    value = float(raw) if "." in raw else int(raw)
                except ValueError:
                    value = raw
            values[key] = value
        return values
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _require_training_modules() -> tuple[Any, Any, Any, Any]:
    try:
        import numpy as np
        from datasets import load_from_disk
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are required. Install requirements-train.txt first."
        ) from exc
    return (
        np,
        load_from_disk,
        (AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments),
        (EarlyStoppingCallback, TrainerCallback),
    )


def _limit(dataset: Any, limit: int | None) -> Any:
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def _load_data(load_from_disk: Any, limit: int | None) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for split in ("train", "val"):
        path = CACHE_DIR / f"primary_{split}"
        if not path.is_dir():
            raise FileNotFoundError(
                f"Missing {path}. Run `python data.py --limit {limit or 200}` first."
            )
        datasets[split] = _limit(load_from_disk(str(path)), limit)
        dataset = datasets[split]
        labels: dict[str, int] = {}
        for label in dataset["label"]:
            labels[str(label)] = labels.get(str(label), 0) + 1
        print(f"primary/{split}: shape=({len(dataset)}, {len(dataset.column_names)})")
        print(f"primary/{split}: label_distribution={labels}")
        for row in dataset.select(range(min(3, len(dataset)))):
            print(json.dumps(row, ensure_ascii=True, default=str))
    return datasets


def _percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _print_token_lengths(tokenizer: Any, datasets: dict[str, Any], max_len: int) -> None:
    lengths: list[int] = []
    for dataset in datasets.values():
        for text in dataset["text"]:
            lengths.append(len(tokenizer(str(text), add_special_tokens=True)["input_ids"]))
    percentiles = {
        str(percentile): _percentile(lengths, percentile / 100)
        for percentile in (50, 90, 95, 99, 100)
    }
    print(f"Token-length percentiles before truncation: {json.dumps(percentiles)}")
    if percentiles["95"] <= max_len:
        print(
            f"max_len={max_len} covers the measured p95 ({percentiles['95']}); "
            "it is a measured tail-coverage cap."
        )
    else:
        print(
            f"max_len={max_len} is below the measured p95 ({percentiles['95']}); "
            "the configured cap will truncate the longest examples."
        )


def _tokenize(datasets: dict[str, Any], tokenizer: Any, max_len: int) -> dict[str, Any]:
    tokenized: dict[str, Any] = {}
    for split, dataset in datasets.items():
        tokenized[split] = dataset.map(
            lambda batch: tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_len,
            ),
            batched=True,
            remove_columns=[column for column in dataset.column_names if column != "label"],
        )
        tokenized[split] = tokenized[split].rename_column("label", "labels")
        tokenized[split].set_format("torch")
    return tokenized


def _metrics_factory(np: Any) -> Any:
    from sklearn.metrics import average_precision_score, accuracy_score, f1_score

    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        logits, labels = eval_prediction
        probabilities = 1 / (1 + np.exp(-logits[:, 1]))
        invalid_probabilities = ~np.isfinite(probabilities)
        if np.any(invalid_probabilities):
            print(
                "WARNING: Non-finite evaluation probabilities detected; "
                "replacing them with 0.5."
            )
            probabilities = np.nan_to_num(
                probabilities,
                nan=0.5,
                posinf=0.5,
                neginf=0.5,
            )
        predictions = (probabilities >= 0.5).astype(int)
        invalid_predictions = ~np.isfinite(predictions)
        if np.any(invalid_predictions):
            print(
                "WARNING: Non-finite evaluation predictions detected; "
                "replacing them with 0."
            )
            predictions = np.nan_to_num(predictions, nan=0, posinf=0, neginf=0).astype(int)
        return {
            "pr_auc": float(average_precision_score(labels, probabilities)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "accuracy": float(accuracy_score(labels, predictions)),
        }

    return compute_metrics


def _arguments(TrainingArguments: Any, config: dict[str, Any], use_fp16: bool) -> Any:
    values = {
        "output_dir": str(ROOT / config["output_dir"]),
        "per_device_train_batch_size": int(config["batch_size"]),
        "per_device_eval_batch_size": int(config["batch_size"]),
        "learning_rate": 1e-5,
        "num_train_epochs": float(config["num_train_epochs"]),
        "warmup_ratio": 0.1,
        "fp16": False,
        "bf16": False,
        "optim": "adamw_torch",
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "seed": int(config["seed"]),
        "data_seed": int(config["seed"]),
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": int(config["logging_steps"]),
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_pr_auc",
        "greater_is_better": True,
        "report_to": [],
        "save_total_limit": 2,
    }
    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in signature:
        values["eval_strategy"] = values.pop("evaluation_strategy")
    return TrainingArguments(**{key: value for key, value in values.items() if key in signature})


def _checkpoint_callback(TrainerCallback: Any, repo_id: str | None) -> Any:
    class EpochCheckpointCallback(TrainerCallback):
        def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            model = kwargs["model"]
            tokenizer = kwargs.get("tokenizer")
            epoch = int(round(state.epoch or 0))
            checkpoint = Path(args.output_dir) / f"checkpoint-epoch-{epoch}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint)
            if tokenizer is not None:
                tokenizer.save_pretrained(checkpoint)
            if repo_id:
                model.push_to_hub(
                    repo_id=repo_id,
                    commit_message=f"Save epoch {epoch} checkpoint",
                )
            return control

    return EpochCheckpointCallback()


def _write_environment() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    )
    (ARTIFACTS_DIR / "env.txt").write_text(result.stdout, encoding="utf-8")


def _write_train_log(trainer: Any) -> None:
    history = trainer.state.log_history
    keys = sorted({key for row in history for key in row})
    with (ARTIFACTS_DIR / "train_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def train(limit: int | None = None, config_path: str | Path | None = None) -> None:
    config = _read_config(config_path)
    repo_id = os.environ.get("HF_REPO")
    if not repo_id:
        raise RuntimeError("HF_REPO is required for training uploads.")
    np, load_from_disk, model_modules, callback_modules = _require_training_modules()
    (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    ) = model_modules
    EarlyStoppingCallback, TrainerCallback = callback_modules
    datasets = _load_data(load_from_disk, limit)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    _print_token_lengths(tokenizer, datasets, int(config["max_len"]))
    tokenized = _tokenize(datasets, tokenizer, int(config["max_len"]))
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=2,
    )
    arguments = _arguments(TrainingArguments, config, False)
    trainer_kwargs = {
        "model": model,
        "args": arguments,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["val"],
        "compute_metrics": _metrics_factory(np),
        "callbacks": [
            _checkpoint_callback(TrainerCallback, repo_id),
            EarlyStoppingCallback(
                early_stopping_patience=int(config["early_stopping_patience"])
            ),
        ],
    }
    try:
        trainer = Trainer(tokenizer=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = Trainer(processing_class=tokenizer, **trainer_kwargs)
    trainer.train()
    best_checkpoint = ARTIFACTS_DIR / "best_checkpoint"
    if best_checkpoint.exists():
        shutil.rmtree(best_checkpoint)
    trainer.save_model(str(best_checkpoint))
    tokenizer.save_pretrained(best_checkpoint)
    trainer.model.push_to_hub(repo_id=repo_id, commit_message="Upload best PromptShield checkpoint")
    tokenizer.push_to_hub(repo_id=repo_id, commit_message="Upload best PromptShield tokenizer")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_train_log(trainer)
    _write_environment()
    print(f"Best checkpoint saved locally to {best_checkpoint}")
    print(f"Best checkpoint pushed to {repo_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML configuration path; defaults to configs/deberta_small.yaml.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows per training split.")
    parser.add_argument("--dry-run", action="store_true", help="Print configuration without loading data or models.")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if args.config is not None and not args.config.is_file():
        parser.error(f"--config file does not exist: {args.config}")
    return args


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print("Dry run: no datasets, models, or GPUs used.")
        print(json.dumps(_read_config(args.config), indent=2))
        print(f"Row limit per split: {args.limit if args.limit is not None else 'unlimited'}")
        return
    train(args.limit, args.config)


if __name__ == "__main__":
    main()
