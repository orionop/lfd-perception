"""
Task segmentation for trials with no gripper topic: group the force-peak
events from force_only_events.py into activity CLUSTERS by time-gap, the
force-only analogue of multi_event.py's grasp/release-bounded cycles.

force_only_events.py already finds every force-magnitude peak above
baseline; on lfdws_t001_depth that's 16 peaks spanning 30-64s (vs. a single
"press" reported when only the strongest peak is used). Without a gripper
topic there's no grasp/release to bound interaction cycles, so instead we
group peaks that are close together in time (gap < --gap_s) into one
activity cluster -- each cluster is a candidate distinct contact/placement
event, not just a monolithic "press".

Reads: the trial's merged CSV directly (same force detection as
force_only_events.py).

Writes:
    figures/identify/events_force_clusters.json
    figures/identify/events_force_clusters.csv

Usage:
    .venv_analysis/bin/python Code/force_only_multi_event.py \
        --trial Data/lfdws_t001_depth --gap_s 3.0
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


def detect_force_peaks(df, force_peak_min_n=5.0, peak_distance_s=0.5):
    """Same logic as force_only_events.py -- returns list of event dicts."""
    t = pd.to_datetime(df[POSE_TS])
    t_rel = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    img_ids = df[IMG].astype(str).to_numpy()

    fm = np.sqrt(df[FX].astype(float) ** 2 + df[FY].astype(float) ** 2 +
                 df[FZ].astype(float) ** 2).to_numpy()
    baseline = float(np.median(fm[: len(fm) // 10]))
    fm_adj = fm - baseline

    dt = float(np.median(np.diff(t_rel)))
    min_dist = max(1, int(round(peak_distance_s / max(dt, 1e-6))))
    peaks, _ = find_peaks(fm_adj, height=force_peak_min_n, distance=min_dist)

    events = []
    for p in peaks:
        p = int(p)
        events.append({"t_rel_s": float(t_rel[p]), "row_idx": p,
                       "img_ts": img_ids[p], "force_mag": float(fm[p])})
    events.sort(key=lambda e: e["t_rel_s"])
    return events


def cluster_by_gap(events, gap_s):
    """Group time-sorted events into clusters separated by > gap_s seconds
    of silence. Each cluster is a candidate distinct contact/activity
    event -- the force-only analogue of a grasp-release interaction cycle."""
    clusters = []
    cur = []
    for e in events:
        if cur and (e["t_rel_s"] - cur[-1]["t_rel_s"]) > gap_s:
            clusters.append(cur)
            cur = []
        cur.append(e)
    if cur:
        clusters.append(cur)
    return clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--gap_s", type=float, default=3.0,
                    help="max seconds between consecutive peaks to stay in "
                         "the same activity cluster")
    ap.add_argument("--force_thr_n", type=float, default=5.0)
    ap.add_argument("--out_json", default="figures/identify/events_force_clusters.json")
    ap.add_argument("--out_csv", default="figures/identify/events_force_clusters.csv")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    csv_path = next((os.path.join(args.trial, f) for f in os.listdir(args.trial)
                     if f.endswith(".csv") and not f.startswith(".")), None)
    print(f"[load] {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    events = detect_force_peaks(df, force_peak_min_n=args.force_thr_n)
    print(f"[detect] {len(events)} force peaks above baseline", flush=True)

    clusters = cluster_by_gap(events, args.gap_s)
    print(f"[cluster] gap_s={args.gap_s} -> {len(clusters)} activity cluster(s)",
          flush=True)
    for i, c in enumerate(clusters):
        t0, t1 = c[0]["t_rel_s"], c[-1]["t_rel_s"]
        peak_mag = max(e["force_mag"] for e in c)
        print(f"  cluster {i+1}: t=[{t0:.2f}, {t1:.2f}]s  "
              f"({len(c)} peaks, max |F|={peak_mag:.1f}N, "
              f"span={t1-t0:.2f}s)  seed_img={c[0]['img_ts']}", flush=True)

    out_doc = {
        "trial": args.trial, "gap_s": args.gap_s,
        "n_peaks": len(events), "n_clusters": len(clusters),
        "clusters": [
            {"cluster_idx": i + 1, "t_start_s": c[0]["t_rel_s"],
             "t_end_s": c[-1]["t_rel_s"], "n_peaks": len(c),
             "max_force_n": max(e["force_mag"] for e in c),
             "events": c}
            for i, c in enumerate(clusters)
        ],
    }
    with open(args.out_json, "w") as f:
        json.dump(out_doc, f, indent=2)
    print(f"[write] {args.out_json}", flush=True)

    with open(args.out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["cluster_idx", "t_rel_s", "row_idx", "img_ts", "force_mag"])
        for i, c in enumerate(clusters):
            for e in c:
                w.writerow([i + 1, e["t_rel_s"], e["row_idx"], e["img_ts"], e["force_mag"]])
    print(f"[write] {args.out_csv}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
