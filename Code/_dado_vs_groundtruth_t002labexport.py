"""
DADO-style label-free proposer vs. proprioception-cued ground truth, on
lfdws_t002_labexport -- the first bag with BOTH real ZED sensor depth AND a
real gripper/force grasp cycle in the same trial. Every earlier DADO
comparison (_dado_real_depth.py, _dado_real_depth_multi.py) ran on
lfdws_t001_depth, which has real depth but NO gripper topic -- there was no
ground-truth object mask to score DADO's proposal against, only qualitative
"does it look noisy" judgment.

Here we have both: Code/propagate_object_n.py already produced a SAM2
ground-truth mask for the grasped Rubik's cube across all 975 frames
(figures/t002labexport/propagation_grasped_summary.csv + per-frame overlays).
This script recovers that mask at the 3 proprioceptive events (grasp, press,
release) and computes IoU against DADO's label-free proposal (DINOv2
attention x real-depth^0.5, thresholded at the 85th percentile -- same
recipe as _dado_real_depth.py) at the same frames. This is the first
quantitative (not just qualitative) DADO-vs-ground-truth number in this
repo.

Output: figures/dado_vs_groundtruth_t002labexport.png (3 rows: RGB w/
ground-truth mask outline | DINOv2 attn | DADO proposal mask, one row per
event) + printed IoU per event.

Run inside .venv_dado:
    .venv_dado/bin/python Code/_dado_vs_groundtruth_t002labexport.py
"""
import csv
import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

TRIAL = "Data/lfdws_t002_labexport/lfdws_t002"
RGB_DIR = f"{TRIAL}/zed_zed_node_rgb_color_rect_image_compressed"
DEPTH_DIR = f"{TRIAL}/zed_zed_node_depth_depth_registered_compressedDepth"
PROPAGATION_SUMMARY = "figures/t002labexport/propagation_grasped_summary.csv"
OUT = "figures/dado_vs_groundtruth_t002labexport.png"

DINO_ID = "facebook/dinov2-base"

# (label, rgb/depth img_id) -- grasp/press/release events from
# Code/multi_event.py on lfdws_t002_labexport (press_5, the peak-force
# press within the grasp-held window)
EVENTS = [
    ("grasp",   "1783420606319992476"),
    ("press",   "1783420610735450476"),
    ("release", "1783420625119277476"),
]


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def dino_attention(pil_img, processor, model, device):
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    attn = out.attentions[-1][0]
    cls_to_patches = attn[:, 0, 1:].mean(0)
    n = int(cls_to_patches.shape[0] ** 0.5)
    a = cls_to_patches.reshape(n, n).cpu().numpy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    return a


def real_depth_closeness(npy_path, H, W):
    depth_m = np.load(npy_path).astype(np.float32)
    valid = depth_m > 0
    if not valid.any():
        return np.zeros((H, W), dtype=np.float32)
    fill = depth_m[valid].max()
    depth_m = np.where(valid, depth_m, fill)
    closeness = 1.0 - (depth_m - depth_m.min()) / (depth_m.max() - depth_m.min() + 1e-9)
    closeness = cv2.resize(closeness, (W, H), interpolation=cv2.INTER_NEAREST)
    return closeness


def dado_mask(a_full, d_full):
    sal = a_full * (d_full ** 0.5)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)
    thr = np.percentile(sal, 85)
    return (sal > thr)


def load_gt_mask(rgb_path, overlay_path, color_bgr=(0, 255, 0), tol=40):
    src = cv2.imread(rgb_path)
    ov = cv2.imread(overlay_path)
    if src is None or ov is None or src.shape != ov.shape:
        return None
    diff = ov.astype(int) - src.astype(int)
    b, g, r = color_bgr
    m = np.ones(diff.shape[:2], dtype=bool)
    for ch, target in zip(range(3), (b, g, r)):
        if target > 100:
            m &= diff[..., ch] > tol
        else:
            m &= diff[..., ch] < tol
    return m


def iou(a, b):
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union) if union > 0 else float("nan")


def to_rgb(arr01):
    g = (arr01 * 255).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(g, cv2.COLORMAP_JET)


def label(img, text):
    ann = img.copy()
    cv2.putText(ann, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 0), 2)
    return ann


def main():
    device = pick_device()
    print(f"[setup] device={device}", flush=True)
    print(f"[load] DINOv2 ({DINO_ID})", flush=True)
    proc = AutoImageProcessor.from_pretrained(DINO_ID)
    model = AutoModel.from_pretrained(DINO_ID, attn_implementation="eager").to(device).eval()

    print(f"[load] propagation ground truth from {PROPAGATION_SUMMARY}", flush=True)
    with open(PROPAGATION_SUMMARY) as f:
        prop_rows = {r["file"].replace(".png", ""): r for r in csv.DictReader(f)}

    rows_out = []
    results = []
    for label_name, img_id in EVENTS:
        rgb_path = f"{RGB_DIR}/{img_id}.png"
        depth_path = f"{DEPTH_DIR}/{img_id}.npy"
        bgr = cv2.imread(rgb_path)
        if bgr is None:
            print(f"  [skip] {label_name}: {rgb_path} missing", flush=True)
            continue
        H, W = bgr.shape[:2]
        rgb_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        print(f"[run] {label_name}: dino attention", flush=True)
        a = dino_attention(rgb_pil, proc, model, device)
        a_full = cv2.resize(a, (W, H), interpolation=cv2.INTER_CUBIC)

        print(f"[run] {label_name}: real depth", flush=True)
        d_full = real_depth_closeness(depth_path, H, W)
        dm = dado_mask(a_full, d_full)

        prop_row = prop_rows.get(img_id)
        gt = None
        if prop_row is not None:
            overlay_path = prop_row["overlay_path"]
            gt = load_gt_mask(rgb_path, overlay_path)
        if gt is None:
            print(f"  [warn] {label_name}: no ground-truth mask (overlay missing/mismatched)",
                  flush=True)
            gt = np.zeros((H, W), dtype=bool)

        score = iou(dm, gt)
        results.append((label_name, score, dm.mean() * 100, gt.mean() * 100))
        print(f"  [result] {label_name}: IoU(DADO, ground_truth)={score:.3f}  "
              f"dado_coverage={dm.mean()*100:.1f}%  gt_coverage={gt.mean()*100:.1f}%",
              flush=True)

        gt_outline = bgr.copy()
        contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(gt_outline, contours, -1, (0, 255, 0), 2)
        red = np.zeros_like(bgr); red[..., 2] = 255
        dado_overlay = np.where(dm[..., None], cv2.addWeighted(bgr, 0.5, red, 0.5, 0), bgr)
        row = np.hstack([
            label(gt_outline, f"{label_name} RGB + GT outline"),
            label(to_rgb(a_full), "DINOv2 attn"),
            label(dado_overlay, f"{label_name} DADO mask  IoU={score:.3f}"),
        ])
        rows_out.append(row)

    if rows_out:
        max_w = max(r.shape[1] for r in rows_out)
        padded = [np.pad(r, ((0, 0), (0, max_w - r.shape[1]), (0, 0))) if r.shape[1] < max_w
                  else r for r in rows_out]
        out = np.vstack(padded)
        os.makedirs("figures", exist_ok=True)
        cv2.imwrite(OUT, out)
        print(f"[save] -> {OUT} ({out.shape[1]}x{out.shape[0]})", flush=True)

    print("\n[summary] event, IoU(DADO, ground_truth), dado_coverage%, gt_coverage%:", flush=True)
    for label_name, score, dc, gc in results:
        print(f"  {label_name:10s}  IoU={score:.3f}  dado={dc:5.1f}%  gt={gc:5.1f}%", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
