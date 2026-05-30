"""
Combine the carrot (green) and cup (magenta) propagated masks into one strip
across the same 6 milestones used for propagation_strip.png.

Reads:
    figures/propagation_summary.csv          (carrot, green overlays already saved)
    figures/propagation_cup_summary.csv      (cup,    magenta overlays already saved)
Writes:
    figures/propagation_both_strip.png
"""
import csv
import os
import sys

import cv2
import numpy as np

CARROT_CSV = "figures/propagation_summary.csv"
CUP_CSV = "figures/propagation_cup_summary.csv"
SRC_IMG_DIR = "Data/lfdws_t001/lfdws_t001/zed_zed_node_rgb_color_rect_image_compressed"
OUT_PATH = "figures/propagation_both_strip.png"

MILESTONES = [
    ("grasp",         1779192188377464163),
    ("post-grasp",    1779192192000000000),
    ("approach",      1779192194000000000),
    ("press",         1779192196405413163),
    ("release",       1779192200620130163),
    ("post-release",  1779192203000000000),
]


def load(csv_path):
    rows = {}
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            ns = int(r["file"].replace(".png", ""))
            rows[ns] = r
    return rows


def overlay_color(img_bgr, mask_bool, color, alpha=0.5):
    if mask_bool is None or mask_bool.sum() == 0:
        return img_bgr
    layer = np.zeros_like(img_bgr)
    layer[mask_bool] = color
    return cv2.addWeighted(img_bgr, 1.0, layer, alpha, 0)


def mask_from_color_overlay(overlay_bgr, src_bgr, color, tol=40):
    """Recover a binary mask from an overlay PNG by colour-distance vs source."""
    diff = overlay_bgr.astype(int) - src_bgr.astype(int)
    target = np.array(color, dtype=int) - 0  # shift expected if alpha=0.5
    # simpler: just threshold where the overlay pixel is close to (img + alpha*color)
    # cheap approx: the channel(s) corresponding to the color are pushed up.
    if color == (0, 255, 0):  # green
        return (overlay_bgr[..., 1].astype(int) - src_bgr[..., 1].astype(int)) > tol
    if color == (255, 0, 255):  # magenta
        b = overlay_bgr[..., 0].astype(int) - src_bgr[..., 0].astype(int)
        r = overlay_bgr[..., 2].astype(int) - src_bgr[..., 2].astype(int)
        return (b > tol) & (r > tol)
    return np.zeros(overlay_bgr.shape[:2], dtype=bool)


def main():
    carrot = load(CARROT_CSV)
    cup = load(CUP_CSV)
    if not carrot and not cup:
        print(f"[fatal] no summary CSVs found", flush=True)
        sys.exit(1)
    print(f"[load] carrot rows: {len(carrot)}  cup rows: {len(cup)}", flush=True)

    available = sorted(set(carrot.keys()) | set(cup.keys()))
    panels = []
    for label, target_ns in MILESTONES:
        nearest = min(available, key=lambda x: abs(x - target_ns))
        fname = f"{nearest}.png"
        src = cv2.imread(os.path.join(SRC_IMG_DIR, fname))
        if src is None:
            print(f"  [skip] {label}: no src {fname}", flush=True)
            continue
        comp = src.copy()
        info_bits = []
        if nearest in carrot:
            ov = cv2.imread(carrot[nearest]["overlay_path"])
            if ov is not None and ov.shape == src.shape:
                m = mask_from_color_overlay(ov, src, (0, 255, 0))
                comp = overlay_color(comp, m, (0, 255, 0))
                info_bits.append(f"carrot={int(m.sum())}px")
        if nearest in cup:
            ov = cv2.imread(cup[nearest]["overlay_path"])
            if ov is not None and ov.shape == src.shape:
                m = mask_from_color_overlay(ov, src, (255, 0, 255))
                comp = overlay_color(comp, m, (255, 0, 255))
                info_bits.append(f"cup={int(m.sum())}px")
        h, w = comp.shape[:2]
        banner = np.zeros((40, w, 3), dtype=np.uint8)
        text = f"{label}  " + "  ".join(info_bits)
        cv2.putText(banner, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2)
        panels.append(np.vstack([banner, comp]))
        print(f"  [pick] {label}: {fname}  {' '.join(info_bits)}", flush=True)

    target_h = 480
    resized = [cv2.resize(p, (int(p.shape[1] * target_h / p.shape[0]), target_h))
               for p in panels]
    strip = np.hstack(resized)
    cv2.imwrite(OUT_PATH, strip)
    print(f"[save] -> {OUT_PATH} ({strip.shape[1]}x{strip.shape[0]})", flush=True)


if __name__ == "__main__":
    main()
