"""
Bbox/centroid-based presence signal, addressing A1 in Docs/FAILURE_MODES.md:
raw mask pixel-count is not a reliable "is the object present" signal
(occlusion/partial-view shrinks a mask that's still correctly tracking the
right object; a mask that's tracking the WRONG thing can also have large
area -- see B4). Bbox diagonal and centroid motion are less sensitive to
partial-occlusion area loss and don't conflate "small mask" with "absent".

Reads objects_summary.csv (as written by build_sidecar.py) -- does not
touch build_sidecar.py or any existing figure. Computes, per object:
  - presence: bbox is valid (not the [-1,-1,-1,-1] sentinel used for
    frames where the object had no mask at all)
  - bbox diagonal (px): sqrt(dx^2 + dy^2) of the bounding box -- more
    stable under partial occlusion than raw area, since a partially-hidden
    object's bbox still roughly spans its true extent as long as ANY part
    is visible at the extremes
  - centroid (bbox midpoint) trajectory, for a rough "did it move" signal

Writes:
    <out>_presence.png   (presence timeline, bbox diagonal, centroid path)
    <out>_presence.csv   (per-frame values)

Usage:
    .venv_analysis/bin/python Code/presence_signal.py \
        --summary_csv figures/identify/objects_summary.csv \
        --out figures/presence_lfdws_t001
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[load] {args.summary_csv}", flush=True)
    rows = []
    with open(args.summary_csv) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f"[load] {len(rows)} rows", flush=True)

    by_obj = {}
    for r in rows:
        by_obj.setdefault((r["obj_id"], r["role"]), []).append(r)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax_diag, ax_area = axes
    csv_out_rows = []
    colors = plt.cm.tab10.colors

    for i, (key, items) in enumerate(sorted(by_obj.items())):
        obj_id, role = key
        items.sort(key=lambda r: int(r["frame_idx"]))
        frames, diag, area, present, cx, cy = [], [], [], [], [], []
        for r in items:
            x0, y0, x1, y1 = (int(r["bbox_x0"]), int(r["bbox_y0"]),
                              int(r["bbox_x1"]), int(r["bbox_y1"]))
            valid = not (x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0)
            frames.append(int(r["frame_idx"]))
            present.append(1 if valid else 0)
            if valid:
                d = float(np.hypot(x1 - x0, y1 - y0))
                diag.append(d)
                cx.append((x0 + x1) / 2)
                cy.append((y0 + y1) / 2)
            else:
                diag.append(np.nan)
                cx.append(np.nan)
                cy.append(np.nan)
            area.append(int(r["mask_px"]))
            csv_out_rows.append([obj_id, role, r["frame_idx"], valid, diag[-1],
                                 cx[-1], cy[-1], area[-1]])

        color = colors[i % len(colors)]
        label = f"obj_id={obj_id} role={role}"
        ax_diag.plot(frames, diag, color=color, label=f"{label} bbox-diagonal", lw=1.3)
        ax_area.plot(frames, area, color=color, label=f"{label} mask-area (px)", lw=1.0, alpha=0.7)

        n_present = sum(present)
        n_total = len(present)
        print(f"  [obj {key}] present {n_present}/{n_total} frames "
              f"({100*n_present/max(n_total,1):.1f}%), "
              f"bbox-diagonal range=[{np.nanmin(diag):.0f}, {np.nanmax(diag):.0f}]px",
              flush=True)

    ax_diag.set_ylabel("bbox diagonal (px)")
    ax_diag.set_title(f"Presence signal (bbox diagonal, less sensitive to "
                      f"partial occlusion) -- {os.path.basename(args.summary_csv)}")
    ax_diag.legend(loc="upper left", fontsize=8)
    ax_diag.grid(True, alpha=0.3)

    ax_area.set_xlabel("frame index")
    ax_area.set_ylabel("mask area (px)")
    ax_area.set_title("Raw mask area (A1's original, noisier under occlusion) -- for comparison")
    ax_area.legend(loc="upper left", fontsize=8)
    ax_area.grid(True, alpha=0.3)

    fig.tight_layout()
    out_png = f"{args.out}_presence.png"
    fig.savefig(out_png, dpi=150)
    print(f"[save] -> {out_png}", flush=True)

    out_csv = f"{args.out}_presence.csv"
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["obj_id", "role", "frame_idx", "present", "bbox_diag_px",
                    "centroid_x", "centroid_y", "mask_px"])
        for row in csv_out_rows:
            w.writerow(row)
    print(f"[save] -> {out_csv}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
