"""
SAM 2's video predictor expects a folder of frames named '00000.jpg', '00001.jpg', ...
Our ZED frames are PNGs named by nanosecond timestamp. This script builds a
parallel folder of .jpg symlinks (then converts if symlink-by-extension is rejected)
in timestamp order, plus an index csv mapping new -> original.

Standalone. Idempotent: if the target folder already has the right number of
files it skips.

Usage:
    python3 prepare_sam2_frames.py --src lfdws_t001/lfdws_t001/zed_zed_node_rgb_color_rect_image_compressed --dst frames_jpg
"""
import argparse
import csv
import os
import sys

import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    src_files = sorted([f for f in os.listdir(args.src) if f.endswith(".png")])
    if not src_files:
        print(f"[fatal] no PNGs in {args.src}", flush=True)
        sys.exit(1)

    os.makedirs(args.dst, exist_ok=True)
    n = len(src_files)
    print(f"[info] converting {n} PNG frames -> JPG in {args.dst}", flush=True)

    pad = max(5, len(str(n - 1)))
    index_rows = []
    for i, fname in enumerate(src_files):
        new_name = f"{i:0{pad}d}.jpg"
        out_path = os.path.join(args.dst, new_name)
        if not os.path.exists(out_path):
            img = cv2.imread(os.path.join(args.src, fname))
            cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        index_rows.append((i, new_name, fname))
        if (i + 1) % 50 == 0 or i == 0 or i == n - 1:
            print(f"  [conv] {i+1}/{n}  {fname} -> {new_name}", flush=True)

    idx_csv = os.path.join(args.dst, "_index.csv")
    with open(idx_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "jpg_name", "original_png"])
        w.writerows(index_rows)
    print(f"[done] wrote {n} frames + index -> {idx_csv}", flush=True)


if __name__ == "__main__":
    main()
