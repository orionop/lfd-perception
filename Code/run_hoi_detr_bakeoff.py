"""Run HOI-DETR on an exported interaction benchmark (GPU machine only)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    bundle = Path(args.bundle).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    out = Path(args.out).resolve()
    benchmark = json.loads((bundle / "benchmark.json").read_text())
    if benchmark.get("purpose") != "inference_only_no_reference_masks":
        raise ValueError("refusing a benchmark that may expose references")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo / "demo"))
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location(
        "hoi_detr_demo", repo / "demo" / "demo.py")
    if spec is None or spec.loader is None:
        raise ImportError("could not load HOI-DETR demo")
    module = importlib.util.module_from_spec(spec)
    old_cwd = Path.cwd()
    try:
        os.chdir(repo)
        spec.loader.exec_module(module)
        module.MODEL_CONFIG = str(
            repo / "projects/configs/co_dino_vit/"
            "co_dino_5scale_vit_large_coco_with_relation_only_all_losses_custom.py")
        module.CHECKPOINT = str(checkpoint)
        module.DEVICE = args.device
        module.INPUT_DIR = str(bundle / "images")
        module.OUTPUT_DIR = str(out)
        module.SCORE_THR = 0.3
        module.NMS_IOU = 0.5
        module.VERBOSE_LABELS = False
        module.EXPORT_JSON = True
        module.main()
    finally:
        os.chdir(old_cwd)

    prediction_path = out / "predictions.json"
    if not prediction_path.exists():
        raise RuntimeError("HOI-DETR did not produce predictions.json")
    predictions = json.loads(prediction_path.read_text())
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True).stdout.strip()
    predictions["bakeoff_provenance"] = {
        "adapter_schema_version": "1.0",
        "benchmark_sha256": sha256(bundle / "benchmark.json"),
        "checkpoint_sha256": sha256(checkpoint),
        "repo_commit": commit,
        "device": args.device,
    }
    temporary = prediction_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(predictions, indent=2))
    os.replace(temporary, prediction_path)
    print(f"[write] {prediction_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
