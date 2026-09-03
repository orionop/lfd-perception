"""
Plot per-frame mask area over time for each tracked object.

Reads the propagation summary CSVs already produced
(propagation_summary.csv = carrot, propagation_cup_summary.csv = cup) and
aligns them on the relative-time axis from the merged demo CSV.

Output:
    figures/mask_area_over_time.png  (or --out)

Standalone — no model inference required.

Events (grasp/press/release vertical markers) are detected from the trial's
own CSV via the same force-only fallback as auto_seed.py / build_sidecar.py,
rather than hardcoded to lfdws_t001's timestamps -- so this generalizes to
trials with no gripper topic (grasp/release just don't get drawn).

Usage:
    .venv_analysis/bin/python Code/mask_area_plot.py \
        --trial Data/lfdws_t001/lfdws_t001 \
        --carrot_csv figures/propagation_summary.csv \
        --cup_csv figures/propagation_cup_summary.csv
"""
import argparse
import ast
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from event_utils import gripper_moved
import pandas as pd

POSE_TS = "NS_1.franka_robot_state_broadcaster.current_pose.timestamp"
GRIP = "NS_1.franka_gripper.joint_states.position"
FX = "bota_post.wrench_body_compensated.wrench.force.x"
FY = "bota_post.wrench_body_compensated.wrench.force.y"
FZ = "bota_post.wrench_body_compensated.wrench.force.z"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"

EVENT_COLORS = {"grasp": "tab:blue", "press": "tab:red", "release": "tab:orange"}


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
    return out, df, t, trel


def parse_gw(c):
    try:
        return float(np.sum(ast.literal_eval(c)))
    except Exception:
        return float("nan")


def detect_event_times(df, trel):
    """Same force-only / gripper-only fallbacks as auto_seed.py --
    returns {name: t_rel_s}."""
    has_force = FX in df.columns
    if has_force:
        fm = np.sqrt(df[FX].astype(float) ** 2 + df[FY].astype(float) ** 2 +
                     df[FZ].astype(float) ** 2).to_numpy()
        baseline = float(np.median(fm[: len(fm) // 10]))
    if GRIP not in df.columns:
        if not has_force:
            return {}
        fm_adj = fm - baseline
        i = int(np.argmax(fm_adj))
        return {"press": float(trel[i])}
    w = df[GRIP].apply(parse_gw).to_numpy()
    w_open, w_closed = float(np.nanmax(w)), float(np.nanmin(w))
    thr = w_closed + 0.5 * (w_open - w_closed)
    # Guard: a gripper that never actuated puts this midpoint INSIDE the
    # sensor's own noise band and manufactures grasp/release out of nothing.
    # Measured on lfdws_t001_labexport: width spans 6.6e-7 m of pure noise yet
    # the unguarded rule reported a grasp at 0.06 s and a release at 7.66 s,
    # which then displaced the contact event from the true 11.15 N peak at
    # 5.08 s to 3.34 s. See Code/event_utils.py.
    closed = (w < thr) if gripper_moved(w) else np.zeros(len(w), dtype=bool)
    cd = np.where((~closed[:-1]) & (closed[1:]))[0] + 1
    cu = np.where((closed[:-1]) & (~closed[1:]))[0] + 1
    out = {}
    if len(cd):
        out["grasp"] = float(trel[int(cd[0])])
    if len(cu):
        out["release"] = float(trel[int(cu[-1])])
    if not has_force:
        return out
    fm_adj = np.where(closed, fm - baseline, -np.inf)
    out["press"] = float(trel[int(np.argmax(fm_adj))])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", default="Data/lfdws_t001/lfdws_t001")
    ap.add_argument("--carrot_csv", default="figures/propagation_summary.csv")
    ap.add_argument("--cup_csv", default="figures/propagation_cup_summary.csv")
    ap.add_argument("--out", default="figures/mask_area_over_time.png")
    args = ap.parse_args()

    demo_csv = next((os.path.join(args.trial, f) for f in os.listdir(args.trial)
                     if f.endswith(".csv") and not f.startswith(".")), None)
    if demo_csv is None:
        raise FileNotFoundError(f"no merged CSV in {args.trial}")

    sources = [("carrot", args.carrot_csv, "tab:green"),
               ("cup", args.cup_csv, "tab:purple")]

    print(f"[load] {demo_csv}", flush=True)
    ts_map, df, t, trel = load_image_ts_to_trel(demo_csv)
    print(f"[load] {len(ts_map)} unique image timestamps -> t_rel mapping", flush=True)
    events = detect_event_times(df, trel)
    print(f"[events] {events}", flush=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, path, color in sources:
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

    for name, ev_t in events.items():
        c = EVENT_COLORS.get(name, "tab:grey")
        ax.axvline(ev_t, color=c, linestyle="--", lw=1.2, alpha=0.7)
        ax.text(ev_t, ax.get_ylim()[1] * 0.97, name, color=c, rotation=90,
                va="top", ha="right", fontsize=9)

    ax.set_xlabel("time (s)")
    ax.set_ylabel("mask area (pixels)")
    ax.set_title(f"Tracked-object mask area over time -- {os.path.basename(args.trial)}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[save] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
