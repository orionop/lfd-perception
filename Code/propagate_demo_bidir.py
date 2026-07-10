"""
Bidirectional propagation for the GRASPED object (carrot), addressing C2 in
Docs/FAILURE_MODES.md: propagate_demo.py only propagates forward from the
grasp event, so frames before the grasp (the reach phase) have no carrot
mask at all. propagate_cup.py already does backward+forward for the
contact-receiver for the same reason; this gives the grasped object the
same treatment.

Standalone script -- does NOT modify propagate_demo.py. Same seed-loading
convention (auto_seeds.csv role=grasped, or hard-coded GRASP_IMG_ID
fallback), same SAM 2 video predictor, same overlay/summary/mp4 output
shape, just propagates both directions from the seed like propagate_cup.py
does.

Usage:
    .venv_sam2/bin/python Code/propagate_demo_bidir.py \
        --trial Data/lfdws_t001/lfdws_t001 --ckpt sam2.1_hiera_large.pt \
        --jpg_dir frames_jpg --out figures/propagation_bidir
"""
import argparse
import csv
import os
import shutil
import sys
import time

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

IMG_DIR_NAME = "zed_zed_node_rgb_color_rect_image_compressed"
JPG_DIR_DEFAULT = "frames_jpg"
GRASP_IMG_ID = 1779192188377464163  # from analyze_demo.py
SEED_POINT_FRAC = (0.7, 0.3)


def backup_if_exists(path):
    if os.path.exists(path):
        bak = path + ".bak"
        shutil.copy2(path, bak)
        print(f"[backup] {path} -> {bak}", flush=True)


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


def list_frames_sorted(img_dir):
    files = [f for f in os.listdir(img_dir) if f.endswith(".png")]
    files.sort()
    return files


def overlay(img_bgr, mask, color=(0, 255, 0), alpha=0.5):
    if mask is None or mask.sum() == 0:
        return img_bgr.copy()
    layer = np.zeros_like(img_bgr)
    layer[mask] = color
    return cv2.addWeighted(img_bgr, 1.0, layer, alpha, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--jpg_dir", default=JPG_DIR_DEFAULT)
    ap.add_argument("--ckpt", default="sam2.1_hiera_large.pt")
    ap.add_argument("--cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--out", default="figures/propagation_bidir")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--auto_seeds_csv", default="figures/identify/auto_seeds.csv")
    ap.add_argument("--offload_video_to_cpu", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    img_dir = os.path.join(args.trial, IMG_DIR_NAME)
    png_frames = list_frames_sorted(img_dir)
    print(f"[setup] {len(png_frames)} source PNG frames in {img_dir}", flush=True)

    if not os.path.isdir(args.jpg_dir):
        print(f"[fatal] {args.jpg_dir} not found — run prepare_sam2_frames.py first",
              flush=True)
        sys.exit(1)
    jpg_frames = sorted([f for f in os.listdir(args.jpg_dir) if f.endswith(".jpg")])
    assert len(jpg_frames) == len(png_frames), \
        f"mismatch: {len(jpg_frames)} jpgs vs {len(png_frames)} pngs"

    probe = cv2.imread(os.path.join(img_dir, png_frames[0]))
    H, W = probe.shape[:2]
    auto = load_auto_seed(args.auto_seeds_csv, "grasped")
    if auto is not None:
        img_id_str, px, py = auto
        seed_name = f"{img_id_str}.png"
        if seed_name not in png_frames:
            print(f"[fatal] auto-seed frame {seed_name} not found", flush=True)
            sys.exit(1)
        seed_idx = png_frames.index(seed_name)
        print(f"[seed-src] AUTO from {args.auto_seeds_csv} (role=grasped)", flush=True)
    else:
        seed_name = f"{GRASP_IMG_ID}.png"
        if seed_name not in png_frames:
            print(f"[fatal] hard-coded seed frame {seed_name} not found", flush=True)
            sys.exit(1)
        seed_idx = png_frames.index(seed_name)
        px = SEED_POINT_FRAC[0] * W
        py = SEED_POINT_FRAC[1] * H
        print(f"[seed-src] HARD-CODED (no {args.auto_seeds_csv})", flush=True)
    print(f"[setup] image {W}x{H}, grasped seed at ({px:.0f},{py:.0f}), frame {seed_idx}",
          flush=True)

    device = pick_device()
    print(f"[load] building SAM 2 video predictor on {device} ...", flush=True)
    t0 = time.time()
    predictor = build_sam2_video_predictor(args.cfg, args.ckpt, device=device)
    print(f"[load] ready in {time.time()-t0:.1f}s", flush=True)

    print(f"[init] init_state on {args.jpg_dir} "
          f"(offload_video_to_cpu={args.offload_video_to_cpu}) ...", flush=True)
    t0 = time.time()
    inference_state = predictor.init_state(
        video_path=args.jpg_dir, offload_video_to_cpu=args.offload_video_to_cpu)
    print(f"[init] ok in {time.time()-t0:.1f}s", flush=True)

    points = np.array([[px, py]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    predictor.add_new_points_or_box(
        inference_state=inference_state, frame_idx=seed_idx,
        obj_id=1, points=points, labels=labels,
    )
    print(f"[seed] grasped object seeded at frame_idx={seed_idx}", flush=True)

    # 1) backward propagation (grasp -> reach start) -- the new coverage
    print("[propagate] phase 1: backward from grasp (reach phase)", flush=True)
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
        cv2.putText(ov, f"f{out_frame_idx:03d} GRASPED px={int(mask.sum())}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        out_path = os.path.join(args.out, f"f{out_frame_idx:04d}_{fname}")
        cv2.imwrite(out_path, ov)
        rows.append((out_frame_idx, fname, int(mask.sum()), out_path))
        n_done += 1
        if n_done % 25 == 0 or n_done == 1:
            print(f"  [back] frame {out_frame_idx:3d} mask={int(mask.sum()):6d}px "
                  f"({n_done} done, {n_done/(time.time()-t0):.2f} f/s)", flush=True)
    print(f"[propagate] backward done — {n_done} frames in {time.time()-t0:.1f}s",
          flush=True)

    # 2) forward propagation (grasp -> release -> end) -- same as propagate_demo.py
    print("[propagate] phase 2: forward from grasp", flush=True)
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
        cv2.putText(ov, f"f{out_frame_idx:03d} GRASPED px={int(mask.sum())}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        out_path = os.path.join(args.out, f"f{out_frame_idx:04d}_{fname}")
        cv2.imwrite(out_path, ov)
        rows.append((out_frame_idx, fname, int(mask.sum()), out_path))
        n_done += 1
        if n_done % 25 == 0 or n_done == 1:
            print(f"  [fwd] frame {out_frame_idx:3d} mask={int(mask.sum()):6d}px "
                  f"({n_done} done, {n_done/(time.time()-t0):.2f} f/s)", flush=True)
    print(f"[propagate] forward done — {n_done} frames in {time.time()-t0:.1f}s",
          flush=True)

    rows.sort(key=lambda r: r[0])
    if rows:
        sample = cv2.imread(rows[0][3])
        h, w = sample.shape[:2]
        mp4_path = f"{args.out}.mp4"
        backup_if_exists(mp4_path)
        writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
        print(f"[video] writing {mp4_path} ({len(rows)} frames @ 15 fps)", flush=True)
        for i, r in enumerate(rows):
            writer.write(cv2.imread(r[3]))
            if (i+1) % 100 == 0:
                print(f"  [video] {i+1}/{len(rows)}", flush=True)
        writer.release()
        print(f"[video] saved -> {mp4_path}", flush=True)

    csv_path = f"{args.out}_summary.csv"
    backup_if_exists(csv_path)
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "file", "mask_px", "overlay_path"])
        for r in rows:
            w.writerow(r)
    print(f"[summary] -> {csv_path}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
