"""
Force-only event detection, for trials with no gripper topic
(e.g. Data/lfdws_t001_depth, extracted via mcap_extract.py -- see
Docs/FAILURE_MODES.md B3).

multi_event.py restricts force-peak search to gripper-held windows, which
needs a grasp/release pair first. With no gripper channel there is no held
window, so this instead runs find_peaks over the WHOLE force-magnitude
trace above baseline. Grasp/release events are simply absent from the
output (reported as empty lists), not guessed at.

Usage:
    .venv_analysis/bin/python Code/force_only_events.py \
        --trial Data/lfdws_t001_depth/lfdws_t001_depth
"""
import argparse
import csv
import json
import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

POSE_TS = "NS_1.franka_robot_state_broadcaster.current_pose.timestamp"
FX = "bota_post.wrench_body_compensated.wrench.force.x"
FY = "bota_post.wrench_body_compensated.wrench.force.y"
FZ = "bota_post.wrench_body_compensated.wrench.force.z"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"


def detect_force_events(df, force_peak_min_n=5.0, peak_distance_s=0.5):
    t = pd.to_datetime(df[POSE_TS])
    t_rel = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    img_ids = df[IMG].astype(str).to_numpy()

    fm = np.sqrt(df[FX].astype(float) ** 2 + df[FY].astype(float) ** 2 +
                 df[FZ].astype(float) ** 2).to_numpy()
    baseline = float(np.median(fm[: len(fm) // 10]))
    fm_adj = fm - baseline

    dt = float(np.median(np.diff(t_rel)))
    min_dist = max(1, int(round(peak_distance_s / max(dt, 1e-6))))

    peaks, props = find_peaks(fm_adj, height=force_peak_min_n, distance=min_dist)

    def pack(idx, name):
        return {"event": name, "t_rel_s": float(t_rel[idx]),
                "row_idx": int(idx), "img_ts": img_ids[idx],
                "force_mag": float(fm[idx]), "force_adj_n": float(fm_adj[idx])}

    events = [pack(int(p), f"press_{i+1}") for i, p in enumerate(peaks)]

    summary = {
        "n_grasps": 0,
        "n_releases": 0,
        "n_force_peaks": int(len(peaks)),
        "force_baseline_n": baseline,
        "note": "no gripper topic in this trial -- grasp/release undetected, "
                "see Docs/FAILURE_MODES.md B3",
    }
    return events, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--out_json", default="figures/identify/events_force_only.json")
    ap.add_argument("--out_csv", default="figures/identify/events_force_only.csv")
    ap.add_argument("--force_thr_n", type=float, default=5.0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    csv_path = next((os.path.join(args.trial, f) for f in os.listdir(args.trial)
                     if f.endswith(".csv") and not f.startswith(".")), None)
    print(f"[load] {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    events, summary = detect_force_events(df, force_peak_min_n=args.force_thr_n)

    print(f"[detect] {summary}", flush=True)
    for e in events:
        print(f"  {e['event']:10s} t={e['t_rel_s']:6.2f}s "
              f"|F|={e['force_mag']:6.2f}N (adj={e['force_adj_n']:6.2f}N) "
              f"img={e['img_ts']}", flush=True)

    with open(args.out_json, "w") as f:
        json.dump({"trial": args.trial, "summary": summary, "events": events},
                   f, indent=2)
    print(f"[write] {args.out_json}", flush=True)

    with open(args.out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["event", "t_rel_s", "row_idx", "img_ts",
                    "force_mag", "force_adj_n"])
        for e in events:
            w.writerow([e["event"], e["t_rel_s"], e["row_idx"], e["img_ts"],
                        e["force_mag"], e["force_adj_n"]])
    print(f"[write] {args.out_csv}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
