"""Prepare one Cog project's pinned model assets for a cloud build."""

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def verify_file(path: Path, asset: dict) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Required model asset is not a regular file: {path.name}")
    if path.stat().st_size != asset["size"]:
        raise RuntimeError(f"Model asset has an unexpected size: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != asset["sha256"]:
        raise RuntimeError(f"Model asset failed SHA-256 verification: {path.name}")


def prepare(model_name: str, verify_only: bool = False) -> None:
    manifest = json.loads((ROOT / "models.json").read_text())
    model = manifest["models"][model_name]
    weights = ROOT / model_name / "weights"
    if weights.is_symlink() or not weights.is_dir():
        raise RuntimeError("The selected project must contain its own weights directory.")
    for filename, asset in model["files"].items():
        target = weights / filename
        if not target.exists():
            if verify_only:
                raise RuntimeError(f"Required model asset is missing: {filename}")
            if not asset["download"]:
                raise RuntimeError(f"Required checked-in inference file is missing: {filename}")
            from huggingface_hub import hf_hub_download

            print(f"Downloading {model['repo_id']}/{filename} at {model['revision']}", flush=True)
            hf_hub_download(
                repo_id=model["repo_id"],
                revision=model["revision"],
                filename=filename,
                local_dir=weights,
                token=os.environ.get("HF_TOKEN") or False,
            )
        verify_file(target, asset)
        print(f"Verified {model_name}/weights/{filename}", flush=True)
    print(f"WEIGHTS_READY {model_name} revision={model['revision']}", flush=True)


def main() -> None:
    models = json.loads((ROOT / "models.json").read_text())["models"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(models))
    parser.add_argument("--verify-only", action="store_true", help="Verify existing local files without downloading anything.")
    args = parser.parse_args()
    prepare(args.model, args.verify_only)


if __name__ == "__main__":
    main()
