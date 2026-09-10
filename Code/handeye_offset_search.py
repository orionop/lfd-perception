"""
Test whether the hand-eye calibration's rejection (3/7, Code/calibrate_hand_eye.py
2026-09-09) is explained by a fixed translational offset between current_pose's
reported origin and the Bota SensONE's true F/T measurement origin, WITHOUT
asking Mark anything -- using only data already in hand.

WHY THIS IS A BOUNDED, PHYSICALLY MOTIVATED SEARCH (not another blind grid)
----------------------------------------------------------------------------
Vlutters' 2026-08-28 email claims current_pose is the Franka tool coordinate
system, configured at the Bota SensONE's "tool mounting side" (the face the
gripper bolts to -- item 2 in bota_sensone_dimensions.png). The sensor's own
F/T coordinate-system origin is not necessarily at that face: the datasheet
drawing gives the sensor body's total robot-mounting-side-to-tool-mounting-side
thickness as 38.00 mm, so if the true F/T origin sits anywhere between the two
faces, the resulting offset is bounded by that physical dimension, not
arbitrary. +/-50mm per axis (comfortably past 38mm to allow for the axis
convention being imperfectly known) is the search bound used below.

WHAT STAYS FIXED, WHAT IS SEARCHED
-----------------------------------
The hand-eye solve's ROTATION (R_he, from calibrate_hand_eye.py's real
2026-09-09 run) is trusted as-is: a rigid offset between two points on the
same end-effector stack does not change the recovered camera orientation,
only its position. Only a translation correction on top of the solved
t_he is searched, since that is exactly what an unmodelled origin offset
would corrupt (see calibrate_hand_eye.py's docstring update).

RIGOR, MATCHING Code/extrinsic_grid_search.py'S DISCIPLINE
-----------------------------------------------------------
1. Leave-one-recording-out cross-validation: the offset is chosen on two
   recordings and scored on the third, which it never saw. The held-out
   number is the one that counts.
2. Empirical null: the full grid is its own null distribution. The best
   score is reported as a percentile of the grid and checked for a sharp
   peak vs. a broad plateau -- a plateau means these 7 events cannot pin
   the offset down, not that any particular value is confirmed.
3. Same wrench_ray.py geometry, same contact_eval_set.py 7 events as every
   other candidate scored in this repo -- no new ground truth invented.

DOES NOT WRITE calibration.yaml or calibration_handeye_result.yaml.

Usage:
    .venv_analysis/bin/python Code/handeye_offset_search.py
"""
import argparse
import csv
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_eval_set import build_events
from wrench_ray import pose_to_T, ray_mask_score, ray_pixels, wrench_line_bota

HANDEYE_RESULT = "calibration_handeye_result.yaml"
CALIB = "calibration.yaml"
OUT_CSV = "figures/handeye_offset_search.csv"
BOTA_SENSOR_THICKNESS_MM = 38.0  # bota_sensone_dimensions.png, item 1 -> 2

S_MIN, S_MAX, S_N = -0.60, 0.60, 240


def load_K(path=CALIB):
    with open(path) as f:
        c = yaml.safe_load(f)
    intr = c["camera_intrinsics"]
    if not intr.get("filled", False):
        print("[fatal] camera_intrinsics is not filled:true", flush=True)
        sys.exit(1)
    return np.array(intr["K"], dtype=float)


def load_handeye(path=HANDEYE_RESULT):
    with open(path) as f:
        d = yaml.safe_load(f)
    T = np.array(d["T_bota_camera"], dtype=float)
    return T[:3, :3], T[:3, 3]


def build_grid(bound_mm, step_mm):
    ax = np.arange(-bound_mm, bound_mm + 1e-9, step_mm) / 1000.0  # -> metres
    dx, dy, dz = np.meshgrid(ax, ax, ax, indexing="ij")
    return np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)


def score_all(R_he, t_he, deltas, events, K):
    """(n_deltas, n_events) hit / centroid-distance arrays."""
    n, m = len(deltas), len(events)
    hit = np.zeros((n, m), dtype=bool)
    dist = np.full((n, m), np.inf, dtype=float)
    for j, ev in enumerate(events):
        r0, fhat = wrench_line_bota(ev["force"], ev["torque"])
        T_base_bota = pose_to_T(*ev["pose"])
        for i, d in enumerate(deltas):
            T_bota_cam = np.eye(4)
            T_bota_cam[:3, :3] = R_he
            T_bota_cam[:3, 3] = t_he + d
            uv, z = ray_pixels(r0, fhat, T_base_bota, T_bota_cam, K,
                               s_min=S_MIN, s_max=S_MAX, n=S_N)
            h, dd, _ = ray_mask_score(uv, ev["mask"], ev["W"], ev["H"])
            hit[i, j], dist[i, j] = h, dd
        print(f"[score] {j+1}/{m} {ev['trial']}/{ev['event']}: "
              f"{int(hit[:, j].sum())}/{n} candidates hit", flush=True)
    return hit, dist


def summarise(name, sel, hit, dist, deltas):
    h, d = hit[:, sel], dist[:, sel]
    rate = h.mean(axis=1)
    md = np.where(np.isfinite(d), d, 1e6).mean(axis=1)
    order = np.lexsort((md, -rate))
    best = order[0]
    pct = float((rate <= rate[best]).mean() * 100.0)
    dm = (deltas[best] * 1000.0).round(1)
    print(f"  [{name}] best hit rate {rate[best]:.3f} "
          f"({int(h[best].sum())}/{len(sel)}), delta=[{dm[0]},{dm[1]},{dm[2]}] mm, "
          f"mean centroid dist {md[best]:.1f} px, "
          f"sits at {pct:.2f}th percentile of the grid", flush=True)
    return best, rate, md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bound_mm", type=float, default=50.0)
    ap.add_argument("--step_mm", type=float, default=5.0)
    args = ap.parse_args()

    print(f"[note] Bota SensONE robot-mounting-to-tool-mounting thickness is "
          f"{BOTA_SENSOR_THICKNESS_MM} mm (bota_sensone_dimensions.png) -- "
          f"the physical scale this search's +/-{args.bound_mm}mm bound is "
          f"meant to bracket.", flush=True)

    R_he, t_he = load_handeye()
    print(f"[load] {HANDEYE_RESULT}: t_he="
          f"{(t_he*1000).round(2).tolist()} mm (rotation held fixed)",
          flush=True)

    events = build_events()
    if len(events) < 4:
        print(f"[fatal] only {len(events)} usable events", flush=True)
        sys.exit(1)
    trials = sorted(set(e["trial"] for e in events))
    print(f"[events] {len(events)} events across {len(trials)} recordings: "
          f"{trials}", flush=True)

    K = load_K()
    deltas = build_grid(args.bound_mm, args.step_mm)
    print(f"[grid] {len(deltas):,} translation-offset candidates "
          f"(+/-{args.bound_mm}mm, {args.step_mm}mm step)", flush=True)

    hit, dist = score_all(R_he, t_he, deltas, events, K)

    print("\n[stage] fit on all events (IN-FOLD, not evidence)", flush=True)
    all_idx = np.arange(len(events))
    best, rate, md = summarise("all", all_idx, hit, dist, deltas)

    print("\n[stage] leave-one-recording-out cross-validation", flush=True)
    loro = []
    for held in trials:
        tr = np.array([i for i, e in enumerate(events) if e["trial"] != held])
        te = np.array([i for i, e in enumerate(events) if e["trial"] == held])
        if len(tr) == 0 or len(te) == 0:
            continue
        b, _, _ = summarise(f"fit-without-{held}", tr, hit, dist, deltas)
        held_rate = float(hit[b, te].mean())
        dm = (deltas[b] * 1000.0).round(1)
        print(f"  [held-out {held}] hit rate {held_rate:.3f} "
              f"({int(hit[b, te].sum())}/{len(te)}) at delta="
              f"[{dm[0]},{dm[1]},{dm[2]}] mm", flush=True)
        loro.append((held, held_rate, dm.tolist()))

    print("\n[stage] is the optimum a peak or a plateau?", flush=True)
    top = np.flatnonzero(rate >= rate[best] - 1e-9)
    print(f"  {len(top):,}/{len(deltas):,} candidates tie at the best hit "
          f"rate ({rate[best]:.3f})", flush=True)
    for k, nm in enumerate(["dx", "dy", "dz"]):
        vals = deltas[top, k] * 1000.0
        print(f"    {nm} spread {vals.min():+.1f} .. {vals.max():+.1f} mm  "
              f"unique={len(np.unique(vals))}", flush=True)

    os.makedirs("figures", exist_ok=True)
    if os.path.exists(OUT_CSV):
        import shutil
        shutil.copy2(OUT_CSV, OUT_CSV + ".bak")
        print(f"[backup] {OUT_CSV} -> {OUT_CSV}.bak", flush=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "dx_mm", "dy_mm", "dz_mm", "hit_rate",
                    "mean_centroid_dist_px"])
        order = np.lexsort((md, -rate))[:200]
        for rank, idx in enumerate(order):
            dm = deltas[idx] * 1000.0
            w.writerow([rank, round(dm[0], 2), round(dm[1], 2),
                        round(dm[2], 2), round(float(rate[idx]), 4),
                        round(float(md[idx]), 2)])
    print(f"[write] {OUT_CSV}", flush=True)
    print("\n[note] calibration.yaml and calibration_handeye_result.yaml "
          "were NOT modified.", flush=True)


if __name__ == "__main__":
    main()
