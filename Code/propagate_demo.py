"""
Step 3 (writeup): Propagate the carrot mask across the demo using SAM 2 video.

- Loads ZED PNG frames from the trial's image folder, in timestamp order.
- Seeds an initial mask using a point prompt on the grasp frame
  (img id 1779192188377464163, the gripper-close event).
- Propagates the mask forward through every subsequent frame.
- Writes per-frame mask overlays into figures/propagation/ and a stitched MP4
  + a summary strip figure spanning the demo.

Live-logs every frame as it's processed.

Usage:
    .venv_sam2/bin/python propagate_demo.py \
        --trial lfdws_t001/lfdws_t001 \
        --ckpt sam2.1_hiera_large.pt
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


def load_auto_seed(csv_path, role):
    """Return (img_id_str, px, py) for the given role from auto_seeds.csv,
    or None if not present."""
    if not os.path.exists(csv_path):
        return None
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("role") == role:
                return (row["img_id"], float(row["seed_x"]), float(row["seed_y"]))
    return None

IMG_DIR_NAME = "zed_zed_node_rgb_color_rect_image_compressed"
JPG_DIR_DEFAULT = "frames_jpg"
GRASP_IMG_ID = 1779192188377464163  # from analyze_demo.py

# Point on the carrot (gripper+carrot enters from upper-right) — same as
# segment_events.py final pick that worked
SEED_POINT_FRAC = (0.7, 0.3)


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def list_frames_sorted(img_dir):
    files = [f for f in os.listdir(img_dir) if f.endswith(".png")]
    # filenames are nanosecond timestamps; lex sort == numeric sort (fixed-width)
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
    ap.add_argument("--jpg_dir", default=JPG_DIR_DEFAULT,
                    help="folder with 00000.jpg ... frames (SAM 2 requirement)")
    ap.add_argument("--ckpt", default="sam2.1_hiera_large.pt")
    ap.add_argument("--cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--out", default="figures/propagation")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--auto_seeds_csv", default="figures/identify/auto_seeds.csv",
                    help="if present, use grasped-role row from this CSV as the seed; "
                         "falls back to hard-coded GRASP_IMG_ID + SEED_POINT_FRAC")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    img_dir = os.path.join(args.trial, IMG_DIR_NAME)
    png_frames = list_frames_sorted(img_dir)
    print(f"[setup] {len(png_frames)} source PNG frames in {img_dir}", flush=True)

    # SAM 2 wants a folder of 00000.jpg, 00001.jpg, ... -- read from --jpg_dir
    if not os.path.isdir(args.jpg_dir):
        print(f"[fatal] {args.jpg_dir} not found — run prepare_sam2_frames.py first",
              flush=True)
        sys.exit(1)
    jpg_frames = sorted([f for f in os.listdir(args.jpg_dir) if f.endswith(".jpg")])
    print(f"[setup] {len(jpg_frames)} jpg frames in {args.jpg_dir}", flush=True)
    assert len(jpg_frames) == len(png_frames), \
        f"mismatch: {len(jpg_frames)} jpgs vs {len(png_frames)} pngs"

    # Resolve seed (auto-CSV if available, else hard-coded fallback)
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
        print(f"[seed-src] AUTO from {args.auto_seeds_csv} (role=grasped)",
              flush=True)
    else:
        seed_name = f"{GRASP_IMG_ID}.png"
        if seed_name not in png_frames:
            print(f"[fatal] hard-coded seed frame {seed_name} not found", flush=True)
            sys.exit(1)
        seed_idx = png_frames.index(seed_name)
        px = SEED_POINT_FRAC[0] * W
        py = SEED_POINT_FRAC[1] * H
        print(f"[seed-src] HARD-CODED (no {args.auto_seeds_csv})", flush=True)
    print(f"[setup] seed = frame {seed_idx}/{len(png_frames)} ({seed_name})",
          flush=True)
    print(f"[setup] image {W}x{H}, seed point ({px:.0f},{py:.0f})", flush=True)

    device = pick_device()
    print(f"[setup] device = {device}", flush=True)

    print(f"[load] building SAM 2 video predictor ({args.cfg}) ...", flush=True)
    t0 = time.time()
    predictor = build_sam2_video_predictor(args.cfg, args.ckpt, device=device)
    print(f"[load] ready in {time.time() - t0:.1f}s", flush=True)

    print(f"[init] initializing inference state on {args.jpg_dir} ...", flush=True)
    t0 = time.time()
    inference_state = predictor.init_state(video_path=args.jpg_dir)
    print(f"[init] ok in {time.time() - t0:.1f}s", flush=True)

    print(f"[seed] adding point prompt at frame_idx={seed_idx}", flush=True)
    points = np.array([[px, py]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)  # foreground
    _, _, _ = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=seed_idx,
        obj_id=1,
        points=points,
        labels=labels,
    )
    print("[seed] done", flush=True)

    print("[propagate] starting forward propagation from seed ...", flush=True)
    n_done = 0
    t0 = time.time()
    summary_rows = []  # (frame_idx, file_name, mask_px, overlay_path)
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state, start_frame_idx=seed_idx
    ):
        # one object
        mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(bool)
        fname = png_frames[out_frame_idx]
        bgr = cv2.imread(os.path.join(img_dir, fname))
        ov = overlay(bgr, mask)
        # mark seed point only on seed frame
        if out_frame_idx == seed_idx:
            cv2.circle(ov, (int(px), int(py)), 8, (0, 0, 255), -1)
        cv2.putText(ov, f"f{out_frame_idx:03d}  {fname}  px={int(mask.sum())}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        out_path = os.path.join(args.out, f"f{out_frame_idx:04d}_{fname}")
        cv2.imwrite(out_path, ov)
        summary_rows.append((out_frame_idx, fname, int(mask.sum()), out_path))
        n_done += 1
        if n_done % 5 == 0 or n_done == 1:
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            print(f"  [prop] frame {out_frame_idx:3d}  mask={int(mask.sum()):6d}px  "
                  f"({n_done} done, {rate:.2f} f/s)", flush=True)

    print(f"[propagate] done — {n_done} frames in {time.time()-t0:.1f}s", flush=True)

    # ----- assemble MP4 from overlays in order -----
    summary_rows.sort(key=lambda r: r[0])
    if summary_rows:
        sample = cv2.imread(summary_rows[0][3])
        h, w = sample.shape[:2]
        mp4_path = os.path.join("figures", "propagation_overlay.mp4")
        # ZED ~15 Hz nominal; use 15 fps
        writer = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
        print(f"[video] writing {mp4_path} ({len(summary_rows)} frames @ 15 fps) ...", flush=True)
        for i, row in enumerate(summary_rows):
            writer.write(cv2.imread(row[3]))
            if (i + 1) % 50 == 0:
                print(f"  [video] {i+1}/{len(summary_rows)} frames written", flush=True)
        writer.release()
        print(f"[video] saved -> {mp4_path}", flush=True)

    # ----- save summary csv -----
    csv_path = os.path.join("figures", "propagation_summary.csv")
    with open(csv_path, "w") as f:
        f.write("frame_idx,file,mask_px,overlay_path\n")
        for r in summary_rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")
    print(f"[summary] saved -> {csv_path}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
