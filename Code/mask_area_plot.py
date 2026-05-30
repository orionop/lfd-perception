"""
Plot per-frame mask area over time for each tracked object.

Reads the propagation summary CSVs already produced
(propagation_summary.csv = carrot, propagation_cup_summary.csv = cup) and
aligns them on the relative-time axis from the merged demo CSV.

Output:
    figures/mask_area_over_time.png

Standalone — no model inference required.
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEMO_CSV = "lfdws_t001/lfdws_t001/lfdws_t001_0.csv"
POSE_TS = "NS_1.franka_robot_state_broadcaster.current_pose.timestamp"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"

SOURCES = [
    ("carrot", "figures/propagation_summary.csv",     "tab:green"),
    ("cup",    "figures/propagation_cup_summary.csv", "tab:purple"),
]

EVENTS = {
    "grasp":   (16.74, "tab:blue"),
    "press":   (24.77, "tab:red"),
    "release": (29.01, "tab:orange"),
}

OUT = "figures/mask_area_over_time.png"


def load_image_ts_to_trel(demo_csv):
    df = pd.read_csv(demo_csv)
    t = pd.to_datetime(df[POSE_TS])
    trel = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    imgs = df[IMG].astype(str).to_numpy()
    # for each unique image id, take the first t_rel where it appears
    out = {}
    for tr, im in zip(trel, imgs):
        if im == "0" or im == "nan":
            continue
        if im not in out:
            out[im] = float(tr)
    return out


def main():
    print(f"[load] {DEMO_CSV}", flush=True)
    ts_map = load_image_ts_to_trel(DEMO_CSV)
    print(f"[load] {len(ts_map)} unique image timestamps -> t_rel mapping", flush=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, path, color in SOURCES:
        if not os.path.exists(path):
            print(f"  [skip] {name}: {path} missing", flush=True)
            continue
        xs, ys = [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                img_id = row["file"].replace(".png", "")
                t_rel = ts_map.get(img_id)
                if t_rel is None:
                    continue
                xs.append(t_rel)
                ys.append(int(row["mask_px"]))
        order = np.argsort(xs)
        xs = np.array(xs)[order]
        ys = np.array(ys)[order]
        ax.plot(xs, ys, color=color, label=f"{name} mask area (px)", lw=1.5)
        print(f"  [plot] {name}: {len(xs)} pts, area max={ys.max() if len(ys) else 0:.0f}",
              flush=True)

    for name, (t, c) in EVENTS.items():
        ax.axvline(t, color=c, linestyle="--", lw=1.2, alpha=0.7)
        ax.text(t, ax.get_ylim()[1] * 0.97, name, color=c, rotation=90,
                va="top", ha="right", fontsize=9)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("mask area (pixels)")
    ax.set_title("Tracked-object mask area over the carrot pick / press / drop demo")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"[save] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
