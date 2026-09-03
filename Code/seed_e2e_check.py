"""
End-to-end check: does the geometric seed reproduce the hand-seeded track?

WHY THE PROXY METRIC IS NOT ENOUGH
----------------------------------
Code/geometric_seed.py reports whether the seed pixel falls inside the
ground-truth mask. That is a proxy, and wrong in both directions: a point just
outside a mask can still propagate the correct object, and a point inside a
drifted mask proves nothing. What actually matters is whether SAM 2 propagates
the same object from the geometric seed as from the hand-picked one.

So this runs the real propagation from the geometric seed and compares the
resulting per-frame mask areas against the existing reference track.

THE COMPARISON DISCIPLINE
-------------------------
Compare mask_px statistics and require the ratio to be about 1. This is the
check that caught the 2026-08-12 audit mistake, where a sidecar was rebuilt
from the superseded point-prompt track (mean 2,350 px) instead of the
box-prompt one (mean 32,522 px) and nothing errored -- the ratio of 0.19 was
the only signal. A filename never says which track is current.

Writes to a DISTINCT --out so no existing artifact is overwritten, per the
repo convention that every propagation output is trial-specific.

Usage:
    .venv_sam2/bin/python Code/seed_e2e_check.py --run
    .venv_analysis/bin/python Code/seed_e2e_check.py --compare_only
"""
import argparse
import csv
import os
import subprocess
import sys

import numpy as np

TRIAL = "Data/lfdws_t001_depth"
ROLE = "contact_receiver"
OBJ_ID = 2
SEED_IMG_ID = "1782835513207923681"          # plate_press event frame
REFERENCE = "figures/propagation_plate_depth_summary.csv"
OUT = "figures/propagation_plate_geoseed"


def stats(path):
    if not os.path.exists(path):
        return None
    px = np.array([float(r["mask_px"]) for r in csv.DictReader(open(path))])
    nz = px[px > 0]
    return {"n": len(px), "n_nonzero": len(nz),
            "mean": float(nz.mean()) if len(nz) else 0.0,
            "median": float(np.median(nz)) if len(nz) else 0.0,
            "max": float(px.max()) if len(px) else 0.0}


def show(name, s):
    if s is None:
        print(f"  {name:34s} MISSING", flush=True)
        return
    print(f"  {name:34s} frames {s['n']:5d}  nonzero {s['n_nonzero']:5d}  "
          f"mean {s['mean']:9.1f}  median {s['median']:9.1f}  "
          f"max {s['max']:9.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--compare_only", action="store_true")
    ap.add_argument("--seed_x", type=float, default=None)
    ap.add_argument("--seed_y", type=float, default=None)
    args = ap.parse_args()

    if args.run:
        if args.seed_x is None or args.seed_y is None:
            print("[fatal] --run needs --seed_x and --seed_y from "
                  "Code/geometric_seed.py", flush=True)
            sys.exit(1)
        cmd = [sys.executable, "Code/propagate_object_n.py",
               "--trial", TRIAL, "--ckpt", "sam2.1_hiera_large.pt",
               "--jpg_dir", "frames_jpg_depth",
               "--obj_id", str(OBJ_ID), "--role", ROLE,
               "--seed_img_id", SEED_IMG_ID,
               "--seed_x", str(args.seed_x), "--seed_y", str(args.seed_y),
               "--offload_video_to_cpu",       # 1013 frames exceed MPS memory
               "--out", OUT]
        print("[run] " + " ".join(cmd), flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"[fatal] propagation exited {r.returncode}", flush=True)
            sys.exit(r.returncode)

    print("\n[compare] geometric-seeded track vs the reference track",
          flush=True)
    ref = stats(REFERENCE)
    new = stats(OUT + "_summary.csv")
    show("reference (hand-seeded)", ref)
    show("geometric seed", new)
    if ref and new and ref["mean"] > 0:
        ratio = new["mean"] / ref["mean"]
        print(f"\n  mean mask_px ratio = {ratio:.3f}", flush=True)
        if 0.7 <= ratio <= 1.4:
            print("  MATCH: the geometric seed reproduces the same object. "
                  "The seeding stage is genuinely replaceable.", flush=True)
        else:
            print("  MISMATCH: the geometric seed propagates a DIFFERENT "
                  "object or a drifted track. Do not adopt it on this "
                  "evidence.", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
