"""
Multi-frame extension of Code/_dado_real_depth.py (which only ran the
real-vs-estimated depth DADO comparison on ONE frame, press_1). This runs
it on all 5 activity-cluster seed frames found by
Code/force_only_multi_event.py on lfdws_t001_depth (see
Docs/FAILURE_MODES.md B5: plate press, screwdriver contact, charger grasp,
charger lift, charger->clamp docking).

Does NOT modify _dado_real_depth.py -- standalone, separate output.

Output: figures/dado_real_depth_multi.png (5 rows: RGB | DINOv2 attn |
real-depth (close=hot) | real-depth mask, one row per event).

Run inside .venv_dado:
    .venv_dado/bin/python Code/_dado_real_depth_multi.py
"""
import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

RGB_DIR = "Data/lfdws_t001_depth/zed_zed_node_rgb_color_rect_image_compressed"
DEPTH_DIR = "Data/lfdws_t001_depth/zed_zed_node_depth_depth_registered_compressedDepth"
OUT = "figures/dado_real_depth_multi.png"

# (label, rgb_img_id, matched_depth_img_id) -- from force_only_multi_event.py's
# 5 clusters + the merged CSV's per-row RGB->depth asof match
EVENTS = [
    ("plate_press",        "1782835513207923681", "1782835513163533696"),
    ("screwdriver_contact", "1782835527086969733", "1782835527053175454"),
    ("charger_grasp",      "1782835537622884551", "1782835537581977576"),
    ("charger_lift",       "1782835540835169730", "1782835540802038251"),
    ("charger_dock",       "1782835545525039497", "1782835545485007420"),
]

DINO_ID = "facebook/dinov2-base"


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


def real_depth_map(png_path, H, W):
    raw = np.array(Image.open(png_path)).astype(np.float32)
    valid = raw > 0
    if not valid.any():
        return np.zeros((H, W), dtype=np.float32), 0.0
    fill = raw[valid].max()
    depth_mm = np.where(valid, raw, fill)
    closeness = 1.0 - (depth_mm - depth_mm.min()) / (depth_mm.max() - depth_mm.min() + 1e-9)
    closeness = cv2.resize(closeness, (W, H), interpolation=cv2.INTER_NEAREST)
    return closeness, float(valid.mean())


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
    dino_proc = AutoImageProcessor.from_pretrained(DINO_ID)
    dino_model = AutoModel.from_pretrained(DINO_ID, attn_implementation="eager").to(device).eval()

    rows_out = []
    coverage_summary = []
    for name, rgb_id, depth_id in EVENTS:
        rgb_path = os.path.join(RGB_DIR, f"{rgb_id}.png")
        depth_path = os.path.join(DEPTH_DIR, f"{depth_id}.png")
        bgr = cv2.imread(rgb_path)
        if bgr is None:
            print(f"  [skip] {name}: missing {rgb_path}", flush=True)
            continue
        rgb_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        H, W = bgr.shape[:2]

        print(f"[run] {name}: dino attention", flush=True)
        a = dino_attention(rgb_pil, dino_proc, dino_model, device)
        a_full = cv2.resize(a, (W, H), interpolation=cv2.INTER_CUBIC)

        print(f"[run] {name}: real depth", flush=True)
        d_full, valid_frac = real_depth_map(depth_path, H, W)

        sal = a_full * (d_full ** 0.5)
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)
        thr = np.percentile(sal, 85)
        mask = (sal > thr).astype(np.uint8) * 255
        coverage = (mask > 0).mean()
        coverage_summary.append((name, valid_frac, coverage))
        print(f"  [result] {name}: depth_valid={valid_frac*100:.1f}%  "
              f"mask_coverage={coverage*100:.1f}%", flush=True)

        green = np.zeros_like(bgr); green[..., 1] = 255
        mask_overlay = np.where(mask[..., None] > 0,
                                cv2.addWeighted(bgr, 0.5, green, 0.5, 0), bgr)
        row = [label(bgr, f"{name} RGB"), label(to_rgb(a_full), "DINOv2 attn"),
               label(to_rgb(d_full), "real depth (close=hot)"),
               label(mask_overlay, f"mask ({coverage*100:.0f}%)")]
        rows_out.append(np.hstack(row))

    if not rows_out:
        print("[fatal] no rows produced", flush=True)
        return

    target_w = min(r.shape[1] for r in rows_out)
    rows_out = [cv2.resize(r, (target_w, int(r.shape[0] * target_w / r.shape[1])))
                for r in rows_out]
    out = np.vstack(rows_out)
    os.makedirs("figures", exist_ok=True)
    cv2.imwrite(OUT, out)
    print(f"\n[save] -> {OUT} ({out.shape[1]}x{out.shape[0]})", flush=True)

    print("\n[summary] event, depth_valid%, mask_coverage%:", flush=True)
    for name, vf, cov in coverage_summary:
        print(f"  {name:22s} depth_valid={vf*100:5.1f}%  mask_coverage={cov*100:5.1f}%",
              flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
