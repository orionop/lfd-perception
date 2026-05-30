"""
Segment the task-relevant object at each interaction event (SAM ViT-H).

Builds on analyze_demo.py: takes the event frames (grasp / press / release) and
runs SAM with a point prompt to segment the object the robot is interacting with.
This turns "object visible at the event frame" into "object segmented".

Prompt point: defaults to a point in the gripper region (upper-centre of frame),
overridable per event via --points. Also dumps SAM's prediction so we can eyeball
mask quality before wiring up end-effector projection as the prompt source.

Usage:
    python3 segment_events.py --trial lfdws_t001/lfdws_t001 \
        --ckpt sam_vit_h_4b8939.pth
"""

import argparse
import ast
import os

import cv2
import numpy as np
import pandas as pd
import torch
from segment_anything import SamPredictor, sam_model_registry

# Event image timestamps (from analyze_demo.py on lfdws_t001)
EVENTS = {
    "grasp": 1779192188377464163,
    "press": 1779192196405413163,
    "release": 1779192200620130163,
}
IMG_DIR = "zed_zed_node_rgb_color_rect_image_compressed"

# Default prompt point as a fraction of (width, height). The gripper enters from
# the top of the ZED frame, object held just below it -> upper-centre.
DEFAULT_POINT_FRAC = (0.5, 0.45)


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def overlay_mask(img_rgb, mask, point, color=(0, 255, 0)):
    out = img_rgb.copy()
    colored = np.zeros_like(out)
    colored[mask] = color
    out = cv2.addWeighted(out, 1.0, colored, 0.5, 0)
    cx, cy = point
    cv2.circle(out, (int(cx), int(cy)), 8, (255, 0, 0), -1)
    cv2.circle(out, (int(cx), int(cy)), 8, (255, 255, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--ckpt", default="sam_vit_h_4b8939.pth")
    ap.add_argument("--model", default="vit_h")
    ap.add_argument("--out", default="figures")
    ap.add_argument(
        "--points",
        default="",
        help="optional per-event override 'grasp:0.5,0.4;press:0.55,0.5' (fractions)",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = pick_device()
    print(f"device: {device}")

    overrides = {}
    if args.points:
        for part in args.points.split(";"):
            name, frac = part.split(":")
            fx, fy = frac.split(",")
            overrides[name.strip()] = (float(fx), float(fy))

    print(f"loading SAM ({args.model}) ...")
    sam = sam_model_registry[args.model](checkpoint=args.ckpt)
    sam.to(device)
    predictor = SamPredictor(sam)

    img_dir = os.path.join(args.trial, IMG_DIR)
    panels = []

    for name, img_id in EVENTS.items():
        path = os.path.join(img_dir, f"{img_id}.png")
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"  [skip] missing {path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        fx, fy = overrides.get(name, DEFAULT_POINT_FRAC)
        point = np.array([[fx * w, fy * h]])
        labels = np.array([1])  # foreground

        predictor.set_image(rgb)
        masks, scores, _ = predictor.predict(
            point_coords=point, point_labels=labels, multimask_output=True
        )
        best = int(np.argmax(scores))
        mask = masks[best]
        print(f"  {name}: score={scores[best]:.3f}  mask_px={int(mask.sum())}")

        ov = overlay_mask(rgb, mask, point[0])
        cv2.putText(ov, f"{name} (score {scores[best]:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        panels.append(ov)

    if panels:
        strip = np.hstack([cv2.resize(p, (p.shape[1] * 480 // p.shape[0], 480)) for p in panels])
        out_path = os.path.join(args.out, "segmented_events.png")
        cv2.imwrite(out_path, cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
