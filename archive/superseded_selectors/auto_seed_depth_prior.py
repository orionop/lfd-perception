"""
Non-EE-projection workaround for auto_seed.py's seed-picking failure
(B4: absolute area cutoff rejects a large correct target; a relative-size
retry regressed by picking the reflective table -- see Docs/FAILURE_MODES.md
C1). This tries a THIRD signal: real per-pixel depth (only available on
trials extracted via mcap_extract.py, e.g. lfdws_t001_depth), added as a
near-camera prior alongside the existing position/size scoring -- not a
replacement for auto_seed.py, a standalone experiment.

Rationale: the press-contact object is, almost by construction, something
the end-effector is touching -- it should be near the gripper in DEPTH as
well as image position. A flat background/table surface at a similar
working-plane depth to the true target may not be separable by depth
alone; this is an empirical test of whether it helps in practice, not a
theoretical guarantee.

Does NOT touch auto_seed.py. Falls back to no depth prior (identical to
existing scoring) automatically whenever no depth frame exists for a
trial -- so this cannot regress lfdws_t001 (which has no depth data at
all; the prior term evaluates to 0 for every candidate, an exact no-op).

Usage:
    .venv_sam2/bin/python Code/auto_seed_depth_prior.py \
        --trial Data/lfdws_t001_depth --ckpt sam_vit_h_4b8939.pth
"""
import argparse
import ast
import csv
import os
import sys

import cv2
import numpy as np

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from event_utils import gripper_moved
import pandas as pd
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

IMG_DIR_NAME = "zed_zed_node_rgb_color_rect_image_compressed"
DEPTH_DIR_NAME = "zed_zed_node_depth_depth_registered_compressedDepth"

POSE_TS = "NS_1.franka_robot_state_broadcaster.current_pose.timestamp"
GRIP = "NS_1.franka_gripper.joint_states.position"
FX = "bota_post.wrench_body_compensated.wrench.force.x"
FY = "bota_post.wrench_body_compensated.wrench.force.y"
FZ = "bota_post.wrench_body_compensated.wrench.force.z"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"
DEPTH_IMG = "zed.zed_node.depth.depth_registered.compressedDepth"


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"  # SamAutomaticMaskGenerator's float64 grids reject MPS


def parse_gw(c):
    try:
        return float(np.sum(ast.literal_eval(c)))
    except Exception:
        return float("nan")


def detect_press_row(df):
    fm = np.sqrt(df[FX].astype(float) ** 2 + df[FY].astype(float) ** 2 +
                 df[FZ].astype(float) ** 2).to_numpy()
    baseline = float(np.median(fm[: len(fm) // 10]))
    if GRIP not in df.columns:
        return int(np.argmax(fm - baseline))
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
    # With no real grasp cycle there is no held window to restrict to, so
    # search the whole recording rather than an all -inf array whose argmax
    # would silently return row 0.
    fm_adj = (np.where(closed, fm - baseline, -np.inf)
              if closed.any() else fm - baseline)
    return int(np.argmax(fm_adj))


def score_mask_with_depth_prior(m, img_shape, depth_mm):
    """Same base scoring as auto_seed.py's score_mask for contact_receiver,
    PLUS a near-camera depth term when depth_mm is available. depth_mm=None
    -> depth term is exactly 0, identical to the original scoring."""
    H, W = img_shape[:2]
    seg = m["segmentation"]
    area = int(seg.sum())
    ys, xs = np.where(seg)
    if len(xs) == 0:
        return -np.inf, None, None
    cx, cy = xs.mean(), ys.mean()

    border = ((seg[0, :].any()) + (seg[-1, :].any()) +
              (seg[:, 0].any()) + (seg[:, -1].any()))
    if border >= 3:
        return -np.inf, (cx, cy), None

    img_area = H * W
    af = area / img_area
    if af < 0.005 or af > 0.4:
        return -np.inf, (cx, cy), None

    horiz = 1.0 - abs(cx - W / 2) / (W / 2)
    vert = cy / H
    size = 1.0 - abs(af - 0.10) * 5

    depth_term = 0.0
    med_depth = None
    if depth_mm is not None:
        valid_px = depth_mm[seg]
        valid_px = valid_px[valid_px > 0]
        if len(valid_px) > 20:
            med_depth = float(np.median(valid_px))
            scene_valid = depth_mm[depth_mm > 0]
            if len(scene_valid) > 100:
                # near-camera prior: closer than the scene median = bonus
                scene_med = float(np.median(scene_valid))
                # normalise: 1.0 if much closer, 0.0 if at/behind scene median
                depth_term = np.clip((scene_med - med_depth) / max(scene_med, 1.0), 0, 1)

    base_score = 0.5 * horiz + 0.3 * vert + 0.2 * max(0.0, size)
    combined = 0.7 * base_score + 0.3 * depth_term
    return combined, (cx, cy), med_depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--ckpt", default="sam_vit_h_4b8939.pth")
    ap.add_argument("--model", default="vit_h")
    ap.add_argument("--out_csv", default="figures/identify/auto_seeds_depth_prior.csv")
    ap.add_argument("--out_overlay", default="figures/identify/auto_seeds_depth_prior.png")
    ap.add_argument("--points_per_side", type=int, default=24)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    csv_path = next((os.path.join(args.trial, f) for f in os.listdir(args.trial)
                     if f.endswith(".csv") and not f.startswith(".")), None)
    img_dir = os.path.join(args.trial, IMG_DIR_NAME)
    depth_dir = os.path.join(args.trial, DEPTH_DIR_NAME)
    has_depth = os.path.isdir(depth_dir)
    print(f"[setup] depth available: {has_depth}", flush=True)

    df = pd.read_csv(csv_path)
    press_row = detect_press_row(df)
    img_id = str(df[IMG].iloc[press_row])
    depth_id = str(df[DEPTH_IMG].iloc[press_row]) if DEPTH_IMG in df.columns else None
    print(f"[load] press row={press_row} img={img_id} depth_img={depth_id}", flush=True)

    depth_mm = None
    if has_depth and depth_id:
        depth_path = os.path.join(depth_dir, f"{depth_id}.png")
        if os.path.exists(depth_path):
            depth_mm = np.array(Image.open(depth_path)).astype(np.float32)
            print(f"[load] depth frame loaded, valid_frac="
                  f"{(depth_mm > 0).mean()*100:.1f}%", flush=True)

    bgr = cv2.imread(os.path.join(img_dir, f"{img_id}.png"))
    H, W = bgr.shape[:2]

    device = pick_device()
    print(f"[load] SAM ({args.model}) on {device}", flush=True)
    sam = sam_model_registry[args.model](checkpoint=args.ckpt).to(device)
    mg = SamAutomaticMaskGenerator(
        sam, points_per_side=args.points_per_side,
        pred_iou_thresh=0.85, stability_score_thresh=0.9,
        min_mask_region_area=400,
    )
    print(f"[gen] SAM auto-mask ({W}x{H}) ...", flush=True)
    masks = mg.generate(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    print(f"[gen] {len(masks)} candidate masks", flush=True)

    best, best_pt, best_mask, best_depth = -np.inf, None, None, None
    for m in masks:
        s, pt, med_depth = score_mask_with_depth_prior(m, bgr.shape, depth_mm)
        if s > best:
            best, best_pt, best_mask, best_depth = s, pt, m, med_depth
    if best_pt is None:
        print("[fatal] no candidate passed scoring", flush=True)
        sys.exit(1)

    cx, cy = best_pt
    mask_px = int(best_mask["segmentation"].sum())
    print(f"[pick] seed=({cx:.0f},{cy:.0f}) score={best:.3f} "
          f"mask_px={mask_px} med_depth_mm={best_depth}", flush=True)

    with open(args.out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["role", "event", "img_id", "seed_x", "seed_y", "mask_px", "med_depth_mm"])
        w.writerow(["contact_receiver", "press", img_id, f"{cx:.1f}", f"{cy:.1f}",
                    mask_px, best_depth])
    print(f"[write] {args.out_csv}", flush=True)

    ov = bgr.copy()
    layer = np.zeros_like(ov); layer[best_mask["segmentation"]] = (255, 0, 255)
    ov = cv2.addWeighted(ov, 1.0, layer, 0.4, 0)
    cv2.circle(ov, (int(cx), int(cy)), 10, (0, 0, 255), -1)
    cv2.circle(ov, (int(cx), int(cy)), 10, (255, 255, 255), 2)
    cv2.putText(ov, f"press -> contact_receiver (depth-prior)", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(args.out_overlay, ov)
    print(f"[write] {args.out_overlay}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
