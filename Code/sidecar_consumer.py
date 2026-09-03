"""Minimal reference consumer for the versioned Deliverable A sidecar."""
from __future__ import annotations

import argparse
import json
from collections import Counter

from validate_sidecar import validate


def load_validated(path: str):
    with open(path) as f:
        data = json.load(f)
    errors, warnings, summary = validate(data, path, require_paths=True)
    if errors:
        raise ValueError("invalid sidecar:\n  " + "\n  ".join(errors))
    return data, warnings, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sidecar")
    args = ap.parse_args()
    data, warnings, summary = load_validated(args.sidecar)
    present = Counter()
    for frame in data["frames"]:
        for obj in frame["objects"]:
            if obj["mask_px"] > 0:
                present[obj["role"]] += 1
    print(json.dumps({"summary": summary, "present_frames_by_role": present,
                      "warnings": warnings}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
