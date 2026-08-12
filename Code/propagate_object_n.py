"""
Generic bidirectional SAM 2 propagation for the Nth object in a trial,
parameterized by --obj_id/--role/--seed (not hardcoded to "carrot" or
"cup" like propagate_demo.py / propagate_cup.py). Written to demonstrate
that the pipeline generalizes past the fixed 2-object (grasped,
contact_receiver) scheme -- see Docs/FAILURE_MODES.md B5, where
lfdws_t001_depth turned out to be a >=4-object task.

Does NOT modify propagate_demo.py / propagate_cup.py. Same SAM 2 video
predictor pattern (backward from seed, then forward), generalized to any
obj_id/role/seed point supplied on the command line -- no CSV lookup,
since a manual seed for a new object is a one-off point, not a role
convention shared across trials.

Usage:
    .venv_sam2/bin/python Code/propagate_object_n.py \
        --trial Data/lfdws_t001_depth --ckpt sam2.1_hiera_large.pt \
        --jpg_dir frames_jpg_depth --obj_id 3 --role tool_contact \
        --seed_img_id 1782835527086969733 --seed_x 480 --seed_y 100 \
        --out figures/propagation_obj3_screwdriver --offload_video_to_cpu
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



# Caption colour. MUST NOT equal any object's mask colour: the sidecar
# builder recovers masks from these overlays by colour-differencing, and a
# caption drawn in the mask colour is recovered as object pixels (phantom
# ~1000px object with a fixed bbox at the caption location). White is safe
# for every role colour in use -- it drives all three channels up, so it
# fails the "off channel must be unchanged" test in event_utils.py.
CAPTION_COLOR = (255, 255, 255)

def backup_if_exists(path):
    if os.path.exists(path):
        bak = path + ".bak"
        shutil.copy2(path, bak)
        print(f"[backup] {path} -> {bak}", flush=True)


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


def overlay(img_bgr, mask, color, alpha=0.5):
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
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--obj_id", type=int, required=True)
    ap.add_argument("--role", required=True)
    ap.add_argument("--seed_img_id", required=True)
    ap.add_argument("--seed_x", type=float, default=None)
    ap.add_argument("--seed_y", type=float, default=None)
    ap.add_argument("--seed_box", default=None,
                    help="x0,y0,x1,y1 box prompt instead of a point. Use for "
                         "multi-colored/multi-part objects where a single "
                         "point is ambiguous (e.g. a Rubik's cube: a point "
                         "on a sticker segments just that sticker)")
    ap.add_argument("--color", default="0,165,255",
                    help="BGR overlay color, comma-separated")
    ap.add_argument("--offload_video_to_cpu", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    color = tuple(int(c) for c in args.color.split(","))

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

    seed_name = f"{args.seed_img_id}.png"
    if seed_name not in png_frames:
        print(f"[fatal] seed frame {seed_name} not found", flush=True)
        sys.exit(1)
    seed_idx = png_frames.index(seed_name)
    box = None
    if args.seed_box:
        box = np.array([float(v) for v in args.seed_box.split(",")], dtype=np.float32)
        px, py = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # for overlay dot
        print(f"[setup] obj_id={args.obj_id} role={args.role} "
              f"seed_box={box.tolist()} frame={seed_idx} ({seed_name})", flush=True)
    else:
        if args.seed_x is None or args.seed_y is None:
            print("[fatal] need --seed_x/--seed_y or --seed_box", flush=True)
            sys.exit(1)
        px, py = args.seed_x, args.seed_y
        print(f"[setup] obj_id={args.obj_id} role={args.role} "
              f"seed=({px:.0f},{py:.0f}) frame={seed_idx} ({seed_name})", flush=True)

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

    if box is not None:
        predictor.add_new_points_or_box(
            inference_state=inference_state, frame_idx=seed_idx,
            obj_id=args.obj_id, box=box,
        )
    else:
        points = np.array([[px, py]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        predictor.add_new_points_or_box(
            inference_state=inference_state, frame_idx=seed_idx,
            obj_id=args.obj_id, points=points, labels=labels,
        )
    print(f"[seed] obj_id={args.obj_id} seeded at frame_idx={seed_idx}", flush=True)

    rows = []

    print("[propagate] phase 1: backward from seed", flush=True)
    n_done = 0
    t0 = time.time()
    for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
        inference_state, start_frame_idx=seed_idx, reverse=True
    ):
        mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(bool)
        fname = png_frames[out_frame_idx]
        bgr = cv2.imread(os.path.join(img_dir, fname))
        ov = overlay(bgr, mask, color)
        if out_frame_idx == seed_idx:
            cv2.circle(ov, (int(px), int(py)), 8, (0, 0, 255), -1)
        cv2.putText(ov, f"f{out_frame_idx:03d} obj{args.obj_id}:{args.role} "
                    f"px={int(mask.sum())}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, CAPTION_COLOR, 2)
        out_path = os.path.join(args.out, f"f{out_frame_idx:04d}_{fname}")
        cv2.imwrite(out_path, ov)
        rows.append((out_frame_idx, fname, int(mask.sum()), out_path))
        n_done += 1
        if n_done % 25 == 0 or n_done == 1:
            print(f"  [back] frame {out_frame_idx:3d} mask={int(mask.sum()):6d}px "
                  f"({n_done} done, {n_done/(time.time()-t0):.2f} f/s)", flush=True)
    print(f"[propagate] backward done — {n_done} frames in {time.time()-t0:.1f}s",
          flush=True)

    print("[propagate] phase 2: forward from seed", flush=True)
    n_done = 0
    t0 = time.time()
    for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
        inference_state, start_frame_idx=seed_idx
    ):
        if out_frame_idx == seed_idx:
            continue
        mask = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze().astype(bool)
        fname = png_frames[out_frame_idx]
        bgr = cv2.imread(os.path.join(img_dir, fname))
        ov = overlay(bgr, mask, color)
        cv2.putText(ov, f"f{out_frame_idx:03d} obj{args.obj_id}:{args.role} "
                    f"px={int(mask.sum())}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, CAPTION_COLOR, 2)
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
