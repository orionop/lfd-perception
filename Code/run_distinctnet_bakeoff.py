"""Run DistinctNet's released motion checkpoint on grasp benchmark pairs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


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
    (out / "masks").mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo))
    from predictor import Predictor  # type: ignore
    import torch
    from PIL import Image
    if not hasattr(Image, "LINEAR"):
        Image.LINEAR = Image.Resampling.BILINEAR

    device = torch.device(args.device)
    records = []
    for case in benchmark["cases"]:
        if case["role"] != "grasped":
            continue
        pair = case["distinctnet"]
        primary = np.array(Image.open(bundle / pair["primary"]).convert("RGB"))
        for variant in ("raw", "stabilized"):
            partner = np.array(Image.open(
                bundle / pair[f"partner_{variant}"]).convert("RGB"))
            predictor = Predictor(str(checkpoint), device=device)
            mask = predictor.predict(primary, partner).astype(np.uint8)
            mask = cv2.resize(mask, (primary.shape[1], primary.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
            relative = Path("masks") / f"{case['case_id']}__{variant}.png"
            if not cv2.imwrite(str(out / relative), mask * 255):
                raise RuntimeError(f"failed to write {relative}")
            records.append({"case_id": case["case_id"], "variant": variant,
                            "mask": relative.as_posix()})

    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True).stdout.strip()
    payload = {
        "schema_version": "1.0",
        "model": "distinctnet_motion_foreground",
        "benchmark_sha256": sha256(bundle / "benchmark.json"),
        "checkpoint_sha256": sha256(checkpoint),
        "repo_commit": commit,
        "device": args.device,
        "predictions": records,
    }
    path = out / "predictions.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"[write] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
