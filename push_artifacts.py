"""Upload the local artifacts directory to a Hugging Face Hub repository."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Upload at most this many files, in deterministic relative-path order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List selected files without uploading them.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    return args


def main() -> None:
    args = parse_args()
    repo_id = os.environ.get("HF_REPO")
    if not repo_id:
        raise RuntimeError("HF_REPO is required, for example: export HF_REPO=org/promptshield")
    if not ARTIFACTS_DIR.is_dir():
        raise FileNotFoundError(f"Artifacts directory does not exist: {ARTIFACTS_DIR}")

    files = sorted(path for path in ARTIFACTS_DIR.rglob("*") if path.is_file())
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        if args.dry_run:
            print("Selected 0 artifact file(s).")
            print("Dry run: no upload performed.")
            return
        raise RuntimeError(f"No files found under {ARTIFACTS_DIR}")
    relative_files = [path.relative_to(ARTIFACTS_DIR).as_posix() for path in files]
    print(f"HF_REPO={repo_id}")
    print(f"Selected {len(relative_files)} artifact file(s):")
    for path in relative_files:
        print(f"  {path}")
    if args.dry_run:
        print("Dry run: no upload performed.")
        return

    try:
        from huggingface_hub import upload_folder
    except ImportError as exc:
        raise RuntimeError(
            "The 'huggingface_hub' package is required. Install requirements-train.txt first."
        ) from exc
    result = upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(ARTIFACTS_DIR),
        allow_patterns=relative_files,
        commit_message="Upload PromptShield artifacts",
    )
    print(f"Uploaded artifacts to {result}")


if __name__ == "__main__":
    main()
