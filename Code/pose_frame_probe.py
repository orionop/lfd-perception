"""
Which frame does current_pose actually publish?

WHY THIS EXISTS
---------------
CLAUDE.md and calibration.yaml both assert that current_pose reports the Bota
SensONE origin rather than the TCP. That assertion is load-bearing: every
attempt to seed the GRASPED role places the gripper centre 104 mm away from
the current_pose origin, per Mark's drawing (104 = 12.28 + 91.72, which his
sheet decodes exactly).

But Code/measure_grasp_offset.py measured, per sample, where the held object
actually sits in the current_pose frame, and got magnitudes of 10-35 mm with
per-trial spreads of 0.3-3.0 mm:

    lfdws_t002_new        [ -8.2, 14.2,  -6.2]
    lfdws_t002_labexport  [ -8.3, 12.0,  -6.7]
    lfdws_t004            [-24.3, 13.3,  19.3]
    lfdws_t005            [-30.3, 19.2,  16.5]

An object held in the fingers cannot be 17 mm from the sensor origin AND
104 mm from it. The tight spreads say this is not measurement noise. Either
the depth back-projection is wrong, or current_pose is not where we think.

WHAT THIS TESTS
---------------
Scores candidate gripper-centre offsets on the only objective that matters:
does the projected point land inside the propagated grasped mask. Candidates:

  zero            current_pose is already at/near the gripper centre
  +-104 mm on
  each axis       the six signed placements Mark's 104 mm allows
  measured means  the per-trial values above, as a positive control -- if
                  these do not score well the scorer itself is broken and no
                  other row means anything

The positive control is the point. A probe that reports failure for every
candidate is indistinguishable from a broken probe unless at least one
candidate is known to be right.

Read only. Writes nothing but stdout.

Usage: .venv_analysis/bin/python Code/pose_frame_probe.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometric_seed import load_calib
from measure_grasp_offset import TRIALS, collect
from wrench_ray import pose_to_T, project_points

CANDIDATES = [
    ("zero (current_pose IS gripper)", [0.0, 0.0, 0.0]),
    ("+104 x", [104.0, 0.0, 0.0]),
    ("-104 x", [-104.0, 0.0, 0.0]),
    ("+104 y", [0.0, 104.0, 0.0]),
    ("-104 y", [0.0, -104.0, 0.0]),
    ("+104 z", [0.0, 0.0, 104.0]),
    ("-104 z", [0.0, 0.0, -104.0]),
    ("measured t002 cluster", [-8.25, 13.1, -6.45]),
    ("measured t004/5 cluster", [-27.3, 16.25, 17.9]),
]


def score(offset_mm, samples, K, T_bc):
    """(n_hits, n_scored, mean px distance to mask centroid over scored)."""
    p = np.array(list(offset_mm), dtype=float) / 1000.0
    hits, scored, dists = 0, 0, []
    for s in samples:
        H, W = s["mask"].shape
        T_bb = pose_to_T(*s["pose"])
        p_base = (T_bb @ np.append(p, 1.0))[:3]
        uv, z = project_points(p_base[None, :], K, T_bb @ T_bc)
        u, v = float(uv[0, 0]), float(uv[0, 1])
        if z[0] <= 0 or not (0 <= u < W and 0 <= v < H):
            continue
        scored += 1
        if s["mask"][int(v), int(u)]:
            hits += 1
        ys, xs = np.nonzero(s["mask"])
        if len(xs):
            dists.append(float(np.hypot(u - xs.mean(), v - ys.mean())))
    return hits, scored, (float(np.mean(dists)) if dists else float("nan"))


def main():
    K, T_bc = load_calib()
    print(f"[calib] bota_to_camera t = "
          f"{(T_bc[:3, 3] * 1000).round(2).tolist()} mm", flush=True)

    samples = []
    for label, tdir, sidecar in TRIALS:
        samples += collect(label, tdir, sidecar, K, T_bc)
    if not samples:
        print("[fatal] no usable grasp samples", flush=True)
        return
    trials = sorted({s["trial"] for s in samples})
    print(f"\n[data] {len(samples)} grasp samples over {len(trials)} trials: "
          f"{trials}\n", flush=True)

    print(f"{'candidate':32s} {'in-mask':>9s} {'in-frame':>9s} "
          f"{'mean dist px':>13s}", flush=True)
    print("-" * 68, flush=True)
    rows = []
    for name, off in CANDIDATES:
        h, n, d = score(off, samples, K, T_bc)
        rows.append((name, h, n, d))
        frac = f"{h}/{n}" if n else "0/0"
        print(f"{name:32s} {frac:>9s} {n:>9d} {d:>13.1f}", flush=True)

    print("\n[read]", flush=True)
    best = max(rows, key=lambda r: (r[1] / r[2]) if r[2] else -1)
    print(f"  best candidate: {best[0]}  ({best[1]}/{best[2]})", flush=True)
    ctrl = [r for r in rows if r[0].startswith("measured")]
    if all(r[2] == 0 or r[1] == 0 for r in ctrl):
        print("  POSITIVE CONTROL FAILED: the measured per-trial offsets do "
              "not land\n  in their own masks, so this probe cannot "
              "distinguish anything.\n  Fix the scorer before reading any "
              "row above.", flush=True)
    else:
        print("  positive control passes, so the other rows are meaningful.",
              flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
