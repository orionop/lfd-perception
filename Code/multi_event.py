"""
Multi-event detection.

`analyze_demo.py:detect_events` returns ONE grasp, ONE release, ONE press --
fine for a single pick-and-place demo, but fails on bags with multiple
interaction cycles (pick A / place A / pick B / place B).

This module returns ALL events of each kind in a single demo:
  - every open->closed gripper transition  -> grasp_k
  - every closed->open gripper transition  -> release_k
  - every local force-magnitude peak above baseline during a held window
                                          -> press_k

Each event still includes (t_rel, row_idx, img_ts). The result groups
events into interaction *cycles*: one cycle = grasp_k ... matching
release_k, plus any presses that fall inside that window.

Usage:
    .venv_analysis/bin/python Code/multi_event.py --trial Data/lfdws_t001/lfdws_t001
"""
import argparse
import ast
import csv
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

POSE_TS = "NS_1.franka_robot_state_broadcaster.current_pose.timestamp"
GRIP = "NS_1.franka_gripper.joint_states.position"
FX = "bota_post.wrench_body_compensated.wrench.force.x"
FY = "bota_post.wrench_body_compensated.wrench.force.y"
FZ = "bota_post.wrench_body_compensated.wrench.force.z"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"


def parse_gw(c):
    try:
        return float(np.sum(ast.literal_eval(c)))
    except Exception:
        return float("nan")


def detect_all_events(df, force_peak_min_n=5.0, peak_distance_s=0.5):
    t = pd.to_datetime(df[POSE_TS])
    t_rel = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    img_ids = df[IMG].astype(str).to_numpy()

    w = df[GRIP].apply(parse_gw).to_numpy()
    w_open, w_closed = float(np.nanmax(w)), float(np.nanmin(w))
    thr = w_closed + 0.5 * (w_open - w_closed)
    closed = w < thr
    cd = (np.where((~closed[:-1]) & (closed[1:]))[0] + 1).tolist()
    cu = (np.where((closed[:-1]) & (~closed[1:]))[0] + 1).tolist()

    fm = np.sqrt(df[FX].astype(float) ** 2 + df[FY].astype(float) ** 2 +
                 df[FZ].astype(float) ** 2).to_numpy()
    baseline = float(np.median(fm[: len(fm) // 10]))
    fm_adj = fm - baseline
    # sampling rate from timestamps
    dt = float(np.median(np.diff(t_rel)))
    min_dist = max(1, int(round(peak_distance_s / max(dt, 1e-6))))

    # restrict force peaks to held windows (between grasp and matching release)
    held_mask = np.zeros_like(fm_adj, dtype=bool)
    for g, r in zip(cd, cu + [len(fm_adj) - 1]):
        if r > g:
            held_mask[g:r] = True
    fm_in_held = np.where(held_mask, fm_adj, -np.inf)
    peaks, _ = find_peaks(fm_in_held, height=force_peak_min_n, distance=min_dist)

    def pack(idx, name):
        return {"event": name, "t_rel_s": float(t_rel[idx]),
                "row_idx": int(idx), "img_ts": img_ids[idx],
                "gripper_w": float(w[idx]),
                "force_mag": float(fm[idx])}

    all_events = []
    for i, k in enumerate(cd):
        all_events.append(pack(k, f"grasp_{i+1}"))
    for i, k in enumerate(cu):
        all_events.append(pack(k, f"release_{i+1}"))
    for i, k in enumerate(peaks):
        all_events.append(pack(int(k), f"press_{i+1}"))
    all_events.sort(key=lambda e: e["t_rel_s"])

    # group into cycles
    cycles = []
    for i, g in enumerate(cd):
        # pair with nearest later release
        r = next((u for u in cu if u > g), len(t_rel) - 1)
        cyc = {
            "cycle_idx": i + 1,
            "grasp": pack(g, "grasp"),
            "release": pack(r, "release"),
            "presses": [pack(int(p), "press") for p in peaks if g <= p <= r],
        }
        cycles.append(cyc)

    summary = {
        "n_grasps": len(cd),
        "n_releases": len(cu),
        "n_force_peaks": int(len(peaks)),
        "n_cycles": len(cycles),
        "force_baseline_n": baseline,
        "gripper_thr_m": float(thr),
        "gripper_open_m": w_open,
        "gripper_closed_m": w_closed,
    }
    return all_events, cycles, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--out_json", default="figures/identify/events_multi.json")
    ap.add_argument("--out_csv", default="figures/identify/events_multi.csv")
    ap.add_argument("--force_thr_n", type=float, default=5.0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    csv_path = next((os.path.join(args.trial, f) for f in os.listdir(args.trial)
                     if f.endswith(".csv") and not f.startswith(".")), None)
    print(f"[load] {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    events, cycles, summary = detect_all_events(df, force_peak_min_n=args.force_thr_n)

    print(f"[detect] {summary}", flush=True)
    for e in events:
        print(f"  {e['event']:11s} t={e['t_rel_s']:6.2f}s "
              f"w={e['gripper_w']:.4f} |F|={e['force_mag']:6.2f}N", flush=True)
    print(f"[cycles] {len(cycles)} interaction cycle(s)", flush=True)
    for c in cycles:
        print(f"  cycle {c['cycle_idx']}: grasp@{c['grasp']['t_rel_s']:.2f}s -> "
              f"release@{c['release']['t_rel_s']:.2f}s "
              f"({len(c['presses'])} press peak(s))", flush=True)

    with open(args.out_json, "w") as f:
        json.dump({"trial": args.trial, "summary": summary,
                   "events": events, "cycles": cycles}, f, indent=2)
    print(f"[write] {args.out_json}", flush=True)

    with open(args.out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["event", "t_rel_s", "row_idx", "img_ts",
                    "gripper_w", "force_mag"])
        for e in events:
            w.writerow([e["event"], e["t_rel_s"], e["row_idx"], e["img_ts"],
                        e["gripper_w"], e["force_mag"]])
    print(f"[write] {args.out_csv}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
