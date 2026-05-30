"""
Step 3 (extension): propagate a SECOND object — the cup — across the demo.

Seeds at the force-contact (press) event, where the cup is clearly visible
under the carrot. Propagates BOTH backward and forward so we get the cup's
mask across the full demo (it's on the table the entire time).

Uses the same .venv_sam2 + SAM 2 video predictor as propagate_demo.py.
Live-logs each frame. Outputs:
    figures/propagation_cup/         per-frame overlays
    figures/propagation_cup.mp4      stitched video
    figures/propagation_cup_summary.csv
    figures/propagation_strip_cup.png    6-milestone strip
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

IMG_DIR_NAME = "zed_zed_node_rgb_color_rect_image_compressed"
PRESS_IMG_ID = 1779192196405413163  # from analyze_demo.py — force-contact event
# Cup sits roughly under the gripper at press; eyeballed from the press frame
SEED_POINT_FRAC_CUP = (0.55, 0.65)


def load_auto_seed(csv_path, role):
    if not os.path.exists(csv_path):
        return None
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("role") == role:
                return (row["img_id"], float(row["seed_x"]), float(row["seed_y"]))
    return None


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def overlay(img_bgr, mask, color=(255, 0, 255), alpha=0.5):
    if mask is None or mask.sum() == 0:
        return img_bgr.copy()
    layer = np.zeros_like(img_bgr)
    layer[mask] = color
    return cv2.addWeighted(img_bgr, 1.0, layer, alpha, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--jpg_dir", default="frames_jpg")
    ap.add_argument("--ckpt", default="sam2.1_hiera_large.pt")
    ap.add_argument("--cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--out", default="figures/propagation_cup")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--auto_seeds_csv", default="figures/identify/auto_seeds.csv",
                    help="if present, use contact_receiver-role row from this CSV "
                         "as the seed; else fall back to PRESS_IMG_ID + SEED_POINT_FRAC_CUP")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    img_dir = os.path.join(args.trial, IMG_DIR_NAME)
    png_frames = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    print(f"[setup] {len(png_frames)} PNG frames", flush=True)
    jpg_frames = sorted([f for f in os.listdir(args.jpg_dir) if f.endswith(".jpg")])
    assert len(jpg_frames) == len(png_frames)

    probe = cv2.imread(os.path.join(img_dir, png_frames[0]))
    H, W = probe.shape[:2]
    auto = load_auto_seed(args.auto_seeds_csv, "contact_receiver")
    if auto is not None:
        img_id_str, px, py = auto
        seed_name = f"{img_id_str}.png"
        if seed_name not in png_frames:
            print(f"[fatal] auto-seed frame {seed_name} not found", flush=True)
            sys.exit(1)
        seed_idx = png_frames.index(seed_name)
        print(f"[seed-src] AUTO from {args.auto_seeds_csv} (role=contact_receiver)",
              flush=True)
    else:
        seed_name = f"{PRESS_IMG_ID}.png"
        if seed_name not in png_frames:
            print(f"[fatal] hard-coded seed frame {seed_name} not found", flush=True)
            sys.exit(1)
        seed_idx = png_frames.index(seed_name)
        px = SEED_POINT_FRAC_CUP[0] * W
        py = SEED_POINT_FRAC_CUP[1] * H
        print(f"[seed-src] HARD-CODED (no {args.auto_seeds_csv})", flush=True)
    print(f"[setup] image {W}x{H}, cup seed at ({px:.0f},{py:.0f}), frame {seed_idx}",
          flush=True)

    device = pick_device()
    print(f"[load] building SAM 2 video predictor on {device} ...", flush=True)
    t0 = time.time()
    predictor = build_sam2_video_predictor(args.cfg, args.ckpt, device=device)
    print(f"[load] ready in {time.time()-t0:.1f}s", flush=True)

    print(f"[init] init_state on {args.jpg_dir} ...", flush=True)
    t0 = time.time()
    inference_state = predictor.init_state(video_path=args.jpg_dir)
    print(f"[init] ok in {time.time()-t0:.1f}s", flush=True)

    points = np.array([[px, py]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=seed_idx,
        obj_id=2,  # different obj_id from carrot (1)
        points=points,
        labels=labels,
    )
    print(f"[seed] cup seeded at frame_idx={seed_idx}", flush=True)

    # 1) backward propagation (press -> grasp -> reach)
    print("[propagate] phase 1: backward from press", flush=True)
    n_done = 0
    t0 = time.time()
    rows = []
    for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
        inference_state, start_frame_idx=seed_idx, reverse=True
    ):
        mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(bool)
        fname = png_frames[out_frame_idx]
        bgr = cv2.imread(os.path.join(img_dir, fname))
        ov = overlay(bgr, mask)
        if out_frame_idx == seed_idx:
            cv2.circle(ov, (int(px), int(py)), 8, (0, 0, 255), -1)
        cv2.putText(ov, f"f{out_frame_idx:03d} CUP px={int(mask.sum())}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        out_path = os.path.join(args.out, f"f{out_frame_idx:04d}_{fname}")
        cv2.imwrite(out_path, ov)
        rows.append((out_frame_idx, fname, int(mask.sum()), out_path))
        n_done += 1
        if n_done % 25 == 0 or n_done == 1:
            print(f"  [back] frame {out_frame_idx:3d} mask={int(mask.sum()):6d}px "
                  f"({n_done} done, {n_done/(time.time()-t0):.2f} f/s)", flush=True)
    print(f"[propagate] backward done — {n_done} frames in {time.time()-t0:.1f}s",
          flush=True)

    # 2) forward propagation (press -> release -> end)
    print("[propagate] phase 2: forward from press", flush=True)
    n_done = 0
    t0 = time.time()
    for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
        inference_state, start_frame_idx=seed_idx
    ):
        if out_frame_idx == seed_idx:
            continue  # already saved during backward
        mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(bool)
        fname = png_frames[out_frame_idx]
        bgr = cv2.imread(os.path.join(img_dir, fname))
        ov = overlay(bgr, mask)
        cv2.putText(ov, f"f{out_frame_idx:03d} CUP px={int(mask.sum())}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        out_path = os.path.join(args.out, f"f{out_frame_idx:04d}_{fname}")
        cv2.imwrite(out_path, ov)
        rows.append((out_frame_idx, fname, int(mask.sum()), out_path))
        n_done += 1
        if n_done % 25 == 0 or n_done == 1:
            print(f"  [fwd]  frame {out_frame_idx:3d} mask={int(mask.sum()):6d}px "
                  f"({n_done} done, {n_done/(time.time()-t0):.2f} f/s)", flush=True)
    print(f"[propagate] forward done — {n_done} frames in {time.time()-t0:.1f}s",
          flush=True)

    rows.sort(key=lambda r: r[0])

    # MP4
    if rows:
        sample = cv2.imread(rows[0][3])
        h, w = sample.shape[:2]
        mp4_path = "figures/propagation_cup.mp4"
        writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
        print(f"[video] writing {mp4_path} ({len(rows)} frames @ 15 fps)", flush=True)
        for i, r in enumerate(rows):
            writer.write(cv2.imread(r[3]))
            if (i+1) % 100 == 0:
                print(f"  [video] {i+1}/{len(rows)}", flush=True)
        writer.release()
        print(f"[video] saved -> {mp4_path}", flush=True)

    csv_path = "figures/propagation_cup_summary.csv"
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "file", "mask_px", "overlay_path"])
        for r in rows:
            w.writerow(r)
    print(f"[summary] -> {csv_path}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
