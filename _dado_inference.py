"""
DADO-style label-free object discovery on the 3 event frames.

We don't bind to the original DADO research repo's API (it lacks a stable
public one); instead we reimplement the published recipe with off-the-shelf
foundation models:

  - DINOv2 (facebook/dinov2-base) for self-attention saliency
  - Depth-Anything-V2 (depth-anything/Depth-Anything-V2-Small-hf) for depth
  - dynamic weighting of attention x depth into a single saliency map
  - simple threshold -> binary proposal mask

Output: figures/dado_events.png (3 frames, each showing
RGB / attention / depth / final mask side-by-side).

This is what gets run by run_dado.py inside .venv_dado.
"""
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (AutoImageProcessor, AutoModel,
                          AutoModelForDepthEstimation)

EVENTS = {
    "grasp":   1779192188377464163,
    "press":   1779192196405413163,
    "release": 1779192200620130163,
}
SRC = "lfdws_t001/lfdws_t001/zed_zed_node_rgb_color_rect_image_compressed"
OUT = "figures/dado_events.png"

DINO_ID = "facebook/dinov2-base"
DEPTH_ID = "depth-anything/Depth-Anything-V2-Small-hf"


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
    # last layer, CLS-to-patches attention, mean over heads
    attn = out.attentions[-1][0]      # (heads, tokens, tokens)
    cls_to_patches = attn[:, 0, 1:].mean(0)
    n = int(cls_to_patches.shape[0] ** 0.5)
    a = cls_to_patches.reshape(n, n).cpu().numpy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    return a


def depth_map(pil_img, processor, model, device):
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    d = out.predicted_depth[0].cpu().numpy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-9)
    return d  # higher value = closer (Depth-Anything convention)


def to_rgb(arr01):
    """grey -> BGR for cv2 hstacking"""
    g = (arr01 * 255).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(g, cv2.COLORMAP_JET)


def main():
    device = pick_device()
    print(f"[setup] device={device}", flush=True)

    print(f"[load] DINOv2 ({DINO_ID})", flush=True)
    dino_proc = AutoImageProcessor.from_pretrained(DINO_ID)
    # eager attention so output_attentions actually returns the matrix
    dino_model = AutoModel.from_pretrained(DINO_ID, attn_implementation="eager").to(device).eval()

    print(f"[load] depth ({DEPTH_ID})", flush=True)
    dep_proc = AutoImageProcessor.from_pretrained(DEPTH_ID)
    dep_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_ID).to(device).eval()

    panels = []
    for name, img_id in EVENTS.items():
        path = os.path.join(SRC, f"{img_id}.png")
        if not os.path.exists(path):
            print(f"  [skip] {name}: missing {path}", flush=True)
            continue
        bgr = cv2.imread(path)
        rgb_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        H, W = bgr.shape[:2]

        print(f"[run] {name}: dino", flush=True)
        a = dino_attention(rgb_pil, dino_proc, dino_model, device)
        a_full = cv2.resize(a, (W, H), interpolation=cv2.INTER_CUBIC)

        print(f"[run] {name}: depth", flush=True)
        d = depth_map(rgb_pil, dep_proc, dep_model, device)
        d_full = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)

        # DADO-style dynamic weighting: combine attention with foreground prior
        # (closer = task-relevant in this fixed-camera setup)
        sal = a_full * (d_full ** 0.5)
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)

        # threshold to a proposal mask (top quartile)
        thr = np.percentile(sal, 85)
        mask = (sal > thr).astype(np.uint8) * 255

        # build a 4-panel row: rgb | attention | depth | masked overlay
        attn_vis = to_rgb(a_full)
        depth_vis = to_rgb(d_full)
        mask_overlay = bgr.copy()
        green = np.zeros_like(bgr); green[..., 1] = 255
        mask_overlay = np.where(mask[..., None] > 0,
                                cv2.addWeighted(bgr, 0.5, green, 0.5, 0),
                                bgr)

        labels = [f"{name} RGB", "DINOv2 attn", "depth (close=hot)",
                  "DADO-style mask"]
        row = []
        for img, lab in zip([bgr, attn_vis, depth_vis, mask_overlay], labels):
            ann = img.copy()
            cv2.putText(ann, lab, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 0), 2)
            row.append(ann)
        panels.append(np.hstack(row))
        print(f"  [done] {name}", flush=True)

    if not panels:
        print("[fatal] no panels", flush=True); sys.exit(1)
    # resize each row to same width then stack
    target_w = min(p.shape[1] for p in panels)
    panels = [cv2.resize(p, (target_w, int(p.shape[0] * target_w / p.shape[1])))
              for p in panels]
    out = np.vstack(panels)
    os.makedirs("figures", exist_ok=True)
    cv2.imwrite(OUT, out)
    print(f"[save] -> {OUT} ({out.shape[1]}x{out.shape[0]})", flush=True)


if __name__ == "__main__":
    main()
