"""
DADO-style label-free proposer, real-depth variant.

A3 in Docs/FAILURE_MODES.md showed the DINOv2-attention x estimated-depth
(Depth-Anything-V2) proposer fails on lfdws_t001 -- attention is noisy,
saliency doesn't isolate objects. That leaves open whether estimated depth
was the weak link. lfdws_t001_depth has real ZED sensor depth (see
mcap_extract.py), so this script reruns the same recipe with REAL depth
substituted for the monocular estimate, on the same force-peak event
(press_1, img 1782835513207923681, matched depth frame 1782835513233382995
-- see Code/force_only_events.py output).

Same DINOv2 attention step; only the depth source changes. If saliency is
still noisy, that confirms the failure is scene/attention-based, not a
depth-estimation artifact. If it's cleaner, real depth was the fix DADO
needed on this rig.

Output: figures/dado_real_depth_compare.png (2 rows: estimated-depth recipe
vs real-depth recipe, same frame, same layout as _dado_inference.py's panels).

Run inside .venv_dado (needs transformers/torch, same as _dado_inference.py):
    .venv_dado/bin/python Code/_dado_real_depth.py
"""
import os

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (AutoImageProcessor, AutoModel,
                          AutoModelForDepthEstimation)

RGB_PATH = "Data/lfdws_t001_depth/zed_zed_node_rgb_color_rect_image_compressed/1782835513207923681.png"
REAL_DEPTH_PATH = "Data/lfdws_t001_depth/zed_zed_node_depth_depth_registered_compressedDepth/1782835513233382995.png"
OUT = "figures/dado_real_depth_compare.png"

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
    attn = out.attentions[-1][0]
    cls_to_patches = attn[:, 0, 1:].mean(0)
    n = int(cls_to_patches.shape[0] ** 0.5)
    a = cls_to_patches.reshape(n, n).cpu().numpy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    return a


def estimated_depth_map(pil_img, processor, model, device):
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    d = out.predicted_depth[0].cpu().numpy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-9)
    return d  # higher = closer


def real_depth_map(png_path, H, W):
    """16-bit mm PNG from mcap_extract.py -> normalized closeness map
    (higher = closer, to match Depth-Anything's convention), invalid
    (0) pixels filled with the max depth in-frame (= farthest = least
    salient) so they don't spuriously read as "close"."""
    raw = np.array(Image.open(png_path)).astype(np.float32)  # mm, 0=invalid
    valid = raw > 0
    if not valid.any():
        return np.zeros((H, W), dtype=np.float32)
    fill = raw[valid].max()
    depth_mm = np.where(valid, raw, fill)
    closeness = 1.0 - (depth_mm - depth_mm.min()) / (depth_mm.max() - depth_mm.min() + 1e-9)
    closeness = cv2.resize(closeness, (W, H), interpolation=cv2.INTER_NEAREST)
    return closeness


def saliency_and_mask(a_full, d_full):
    sal = a_full * (d_full ** 0.5)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)
    thr = np.percentile(sal, 85)
    mask = (sal > thr).astype(np.uint8) * 255
    return sal, mask


def to_rgb(arr01):
    g = (arr01 * 255).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(g, cv2.COLORMAP_JET)


def label(img, text):
    ann = img.copy()
    cv2.putText(ann, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 0), 2)
    return ann


def build_row(bgr, a_full, d_full, mask, tag):
    depth_vis = to_rgb(d_full)
    green = np.zeros_like(bgr); green[..., 1] = 255
    mask_overlay = np.where(mask[..., None] > 0,
                            cv2.addWeighted(bgr, 0.5, green, 0.5, 0), bgr)
    row = [label(bgr, f"{tag} RGB"),
           label(to_rgb(a_full), "DINOv2 attn"),
           label(depth_vis, f"{tag} depth (close=hot)"),
           label(mask_overlay, f"{tag} mask")]
    return np.hstack(row)


def main():
    device = pick_device()
    print(f"[setup] device={device}", flush=True)

    print(f"[load] DINOv2 ({DINO_ID})", flush=True)
    dino_proc = AutoImageProcessor.from_pretrained(DINO_ID)
    dino_model = AutoModel.from_pretrained(DINO_ID, attn_implementation="eager").to(device).eval()

    print(f"[load] estimated depth ({DEPTH_ID})", flush=True)
    dep_proc = AutoImageProcessor.from_pretrained(DEPTH_ID)
    dep_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_ID).to(device).eval()

    bgr = cv2.imread(RGB_PATH)
    if bgr is None:
        raise FileNotFoundError(RGB_PATH)
    rgb_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    H, W = bgr.shape[:2]

    print("[run] dino attention", flush=True)
    a = dino_attention(rgb_pil, dino_proc, dino_model, device)
    a_full = cv2.resize(a, (W, H), interpolation=cv2.INTER_CUBIC)

    print("[run] estimated depth (Depth-Anything-V2)", flush=True)
    d_est = estimated_depth_map(rgb_pil, dep_proc, dep_model, device)
    d_est_full = cv2.resize(d_est, (W, H), interpolation=cv2.INTER_CUBIC)
    sal_est, mask_est = saliency_and_mask(a_full, d_est_full)
    row_est = build_row(bgr, a_full, d_est_full, mask_est, "est")
    print(f"  est mask coverage: {(mask_est > 0).mean()*100:.1f}% of frame", flush=True)

    print("[run] real depth (ZED sensor, via mcap_extract.py)", flush=True)
    d_real_full = real_depth_map(REAL_DEPTH_PATH, H, W)
    sal_real, mask_real = saliency_and_mask(a_full, d_real_full)
    row_real = build_row(bgr, a_full, d_real_full, mask_real, "real")
    print(f"  real mask coverage: {(mask_real > 0).mean()*100:.1f}% of frame", flush=True)

    out = np.vstack([row_est, row_real])
    os.makedirs("figures", exist_ok=True)
    cv2.imwrite(OUT, out)
    print(f"[save] -> {OUT} ({out.shape[1]}x{out.shape[0]})", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
