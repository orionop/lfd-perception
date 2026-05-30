"""
Demo analysis for proprioception-cued object grounding (RLBD / UTwente).

Reads the merged CSV from a trial, detects interaction events from gripper and
force signals, plots the proprioceptive timeline, and pulls the ZED frames at
each event so we can verify the task-relevant object is visible at that moment.

Usage:
    python3 analyze_demo.py --trial lfdws_t001/lfdws_t001
"""

import argparse
import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POSE_TS = "NS_1.franka_robot_state_broadcaster.current_pose.timestamp"
GRIP = "NS_1.franka_gripper.joint_states.position"
FX = "bota_post.wrench_body_compensated.wrench.force.x"
FY = "bota_post.wrench_body_compensated.wrench.force.y"
FZ = "bota_post.wrench_body_compensated.wrench.force.z"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"
IMG_DIR = "zed_zed_node_rgb_color_rect_image_compressed"


def parse_gripper_width(cell):
    """Gripper column is a string like '[0.031, 0.031]'; width = sum of fingers."""
    try:
        fingers = ast.literal_eval(cell)
        return float(np.sum(fingers))
    except (ValueError, SyntaxError, TypeError):
        return np.nan


def load(trial_dir):
    csv_path = None
    for f in os.listdir(trial_dir):
        if f.endswith(".csv"):
            csv_path = os.path.join(trial_dir, f)
            break
    if csv_path is None:
        raise FileNotFoundError(f"No merged CSV in {trial_dir}")

    df = pd.read_csv(csv_path)
    t = pd.to_datetime(df[POSE_TS])
    df["t_rel"] = (t - t.iloc[0]).dt.total_seconds()
    df["grip_w"] = df[GRIP].apply(parse_gripper_width)
    df["force_mag"] = np.sqrt(
        df[FX].astype(float) ** 2 + df[FY].astype(float) ** 2 + df[FZ].astype(float) ** 2
    )
    return df


def detect_events(df):
    """Return dict of event_name -> (t_rel, row_index)."""
    events = {}

    # Gripper width: high = open, low = closed. Use derivative for transitions.
    w = df["grip_w"].to_numpy()
    dw = np.gradient(w)

    close_idx = int(np.argmin(dw))   # sharpest narrowing -> grasp
    open_idx = int(np.argmax(dw))    # sharpest widening  -> release
    events["gripper_close"] = (df["t_rel"].iloc[close_idx], close_idx)
    events["gripper_open"] = (df["t_rel"].iloc[open_idx], open_idx)

    # Force contact: baseline-subtracted magnitude peak (the press).
    fm = df["force_mag"].to_numpy()
    baseline = np.median(fm[: len(fm) // 10])  # first 10% as rest baseline
    contact_idx = int(np.argmax(fm - baseline))
    events["force_contact"] = (df["t_rel"].iloc[contact_idx], contact_idx)

    return events


def plot_timeline(df, events, out_path):
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(df["t_rel"], df["grip_w"], color="tab:blue", label="gripper width (m)")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("gripper width (m)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(df["t_rel"], df["force_mag"], color="tab:red", alpha=0.7, label="|force| (N)")
    ax2.set_ylabel("|force| (N)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    colors = {"gripper_close": "green", "force_contact": "purple", "gripper_open": "orange"}
    for name, (t, _) in events.items():
        ax1.axvline(t, color=colors.get(name, "gray"), linestyle="--", linewidth=1.5)
        ax1.text(t, ax1.get_ylim()[1], name, rotation=90, va="top", ha="right", fontsize=8)

    plt.title("Proprioceptive timeline — pick / press / drop")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved timeline -> {out_path}")


def save_event_frames(df, events, trial_dir, out_path):
    img_folder = os.path.join(trial_dir, IMG_DIR)
    n = len(events)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, (t, idx)) in zip(axes, events.items()):
        img_id = str(df[IMG].iloc[idx])
        frame_path = os.path.join(img_folder, f"{img_id}.png")
        if os.path.exists(frame_path):
            ax.imshow(plt.imread(frame_path))
        else:
            ax.text(0.5, 0.5, f"missing\n{img_id}", ha="center", va="center")
        ax.set_title(f"{name}\nt={t:.2f}s")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved event frames -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True, help="trial dir containing merged CSV + image folder")
    ap.add_argument("--out", default="figures", help="output dir for figures")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load(args.trial)
    events = detect_events(df)

    print(f"loaded {len(df)} rows, {df['t_rel'].iloc[-1]:.1f}s demo")
    for name, (t, idx) in events.items():
        print(f"  {name:15s} t={t:6.2f}s  width={df['grip_w'].iloc[idx]:.4f}  "
              f"|F|={df['force_mag'].iloc[idx]:.2f}N  img={df[IMG].iloc[idx]}")

    plot_timeline(df, events, os.path.join(args.out, "timeline.png"))
    save_event_frames(df, events, args.trial, os.path.join(args.out, "event_frames.png"))


if __name__ == "__main__":
    main()
