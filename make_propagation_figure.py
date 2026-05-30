"""
Build a single comparison figure: 6 milestone overlays spanning the demo
(grasp -> mid-move -> approach -> press -> release -> post-release).

Run after propagate_demo.py.

Usage:
    .venv_sam2/bin/python make_propagation_figure.py
"""
import csv
import os
import sys

import cv2
import numpy as np

SUMMARY_CSV = "figures/propagation_summary.csv"
OUT_PATH = "figures/propagation_strip.png"

# milestones from analyze_demo.py event times (ns timestamps), plus mid-points
MILESTONES = [
    ("grasp",          1779192188377464163),
    ("post-grasp",     1779192192000000000),
    ("approach",       1779192194000000000),
    ("press",          1779192196405413163),
    ("release",        1779192200620130163),
    ("post-release",   1779192203000000000),
]


def main():
    if not os.path.exists(SUMMARY_CSV):
        print(f"[fatal] {SUMMARY_CSV} missing — run propagate_demo.py first", flush=True)
        sys.exit(1)

    rows = []
    with open(SUMMARY_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    print(f"[load] {len(rows)} summary rows", flush=True)

    # map ns_timestamp -> row
    by_ns = {int(r["file"].replace(".png", "")): r for r in rows}
    available_ns = sorted(by_ns.keys())

    panels = []
    for label, target_ns in MILESTONES:
        # nearest available
        nearest = min(available_ns, key=lambda x: abs(x - target_ns))
        row = by_ns[nearest]
        img = cv2.imread(row["overlay_path"])
        if img is None:
            print(f"  [skip] {label}: cannot read {row['overlay_path']}", flush=True)
            continue
        # banner at top
        h, w = img.shape[:2]
        banner = np.zeros((40, w, 3), dtype=np.uint8)
        cv2.putText(banner, f"{label} (f{row['frame_idx']}  {int(row['mask_px'])}px)",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        panel = np.vstack([banner, img])
        panels.append(panel)
        print(f"  [pick] {label}: frame {row['frame_idx']} (mask {row['mask_px']}px)", flush=True)

    if not panels:
        print("[fatal] no panels selected", flush=True)
        sys.exit(1)

    target_h = 480
    resized = []
    for p in panels:
        ph, pw = p.shape[:2]
        new_w = int(pw * target_h / ph)
        resized.append(cv2.resize(p, (new_w, target_h)))
    strip = np.hstack(resized)
    cv2.imwrite(OUT_PATH, strip)
    print(f"[save] -> {OUT_PATH} ({strip.shape[1]}x{strip.shape[0]})", flush=True)


if __name__ == "__main__":
    main()
