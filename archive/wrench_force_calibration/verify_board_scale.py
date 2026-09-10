"""
Cross-check the ChArUco board's assumed physical size (--square_size_m) against
the ZED's own independently-measured depth, using data already in hand -- no
new recording, no further confirmation from Mark needed.

WHY THIS MATTERS
----------------
solvePnP's recovered board depth (tvec[2]) is NOT an independent measurement:
it is entirely a function of the ASSUMED object size (square_size_m/
marker_size_m) and the observed pixel geometry. If Mark's printer scaled the
board when printing (a "fit to page" default, a driver quirk, anything other
than literal 100%/actual-size), solvePnP will silently compute a
self-consistent but WRONG depth and pose for every single frame -- exactly
the "tight internal residual, wrong external result" symptom the 2026-09-09
and 2026-09-10 hand-eye attempts both showed, and exactly the one major
assumption in this pipeline that has never been independently checked (we
verified the SOURCE PDF is exact A4/35mm via pdfinfo, but never verified the
PHYSICAL PRINTOUT is exact -- that step depends on Mark's print settings,
which we don't control and never asked him to confirm).

The ZED's own stereo depth is a genuinely independent measurement -- it does
not depend on square_size_m at all. Comparing it against solvePnP's recovered
depth at the same instant directly tests the scale assumption:

    true_square_size_m = assumed_square_size_m * (zed_depth / solvepnp_depth)

If the ratio is ~1.00, the board is the size we assumed. If it's not, this
gives the actual physical square size to use, not just a yes/no answer.

Usage:
    .venv_analysis/bin/python Code/verify_board_scale.py \
        --trial Data/charuco_calib_002 --square_size_m 0.035 --marker_size_m 0.026 \
        --squares_x 5 --squares_y 7 --max_speed_mps 0.001 --n_samples 15
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_hand_eye import (IMG, IMG_DIR_NAME, PX, PY, PZ, QW, QX, QY, QZ,
                                POSE_TS, detect_board_pose, load_calibration)

DEPTH_DIR_NAME = "zed_zed_node_depth_depth_registered_compressedDepth"
DEPTH_COL = "zed.zed_node.depth.depth_registered.compressedDepth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--squares_x", type=int, default=5)
    ap.add_argument("--squares_y", type=int, default=7)
    ap.add_argument("--square_size_m", type=float, required=True)
    ap.add_argument("--marker_size_m", type=float, required=True)
    ap.add_argument("--max_speed_mps", type=float, default=None)
    ap.add_argument("--n_samples", type=int, default=15,
                    help="max number of well-detected frames to check")
    ap.add_argument("--patch_radius_px", type=int, default=8,
                    help="median depth over a small patch, not a single pixel")
    args = ap.parse_args()

    calib = load_calibration()
    K = np.array(calib["camera_intrinsics"]["K"], dtype=float)
    dist = np.array(calib["camera_intrinsics"]["dist"], dtype=float)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y), args.square_size_m,
        args.marker_size_m, aruco_dict)

    demo_csv = next((os.path.join(args.trial, f) for f in os.listdir(args.trial)
                     if f.endswith(".csv") and not f.startswith(".")), None)
    df = pd.read_csv(demo_csv)
    print(f"[load] {demo_csv} ({len(df)} rows)", flush=True)

    speed = None
    if args.max_speed_mps is not None:
        t_s = df[POSE_TS].to_numpy(dtype=float) / 1e9
        pos = df[[PX, PY, PZ]].to_numpy(dtype=float)
        dt = np.diff(t_s)
        dp = np.diff(pos, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.linalg.norm(dp, axis=1) / np.where(dt > 0, dt, np.nan)
        speed = np.full(len(df), np.nan)
        speed[1:] = v
        speed[0] = v[0] if len(v) else np.nan

    img_dir = os.path.join(args.trial, IMG_DIR_NAME)
    depth_dir = os.path.join(args.trial, DEPTH_DIR_NAME)

    ratios = []
    n_checked = 0
    seen_img_ids = set()
    for i, r in df.iterrows():
        if len(ratios) >= args.n_samples:
            break
        if speed is not None:
            s = speed[i]
            if not np.isfinite(s) or s >= args.max_speed_mps:
                continue
        img_id = str(r[IMG])
        if img_id in seen_img_ids:
            # the merged CSV repeats each image across many current_pose
            # rows (current_pose ~1kHz, camera ~15fps) -- without this,
            # "15 samples" would silently be the same handful of images
            # re-measured, not 15 independent checks.
            continue
        seen_img_ids.add(img_id)
        img_path = os.path.join(img_dir, f"{img_id}.png")
        if not os.path.exists(img_path):
            continue
        depth_id = str(r[DEPTH_COL])
        depth_path = os.path.join(depth_dir, f"{depth_id}.png")
        if not os.path.exists(depth_path):
            continue
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        det = detect_board_pose(bgr, board, K, dist)
        if det is None:
            continue
        n_checked += 1
        rvec, tvec = det
        solvepnp_depth_m = float(tvec.reshape(-1)[2])

        # project the board's own origin (0,0,0) to a pixel, and read real
        # depth in a small patch there -- independent of square_size_m
        uv, _ = cv2.projectPoints(np.zeros((1, 3)), rvec, tvec, K, dist)
        u, v = uv.reshape(2)
        depth_png = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_png is None:
            continue
        h, w = depth_png.shape[:2]
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < w and 0 <= vi < h):
            continue
        r0, r1 = max(0, vi - args.patch_radius_px), min(h, vi + args.patch_radius_px + 1)
        c0, c1 = max(0, ui - args.patch_radius_px), min(w, ui + args.patch_radius_px + 1)
        patch = depth_png[r0:r1, c0:c1].astype(np.float64)
        valid = patch[patch > 0]
        if valid.size < 5:
            print(f"  [skip] {img_id}: too few valid depth px in patch "
                  f"({valid.size})", flush=True)
            continue
        zed_depth_mm = float(np.median(valid))
        zed_depth_m = zed_depth_mm / 1000.0

        ratio = zed_depth_m / solvepnp_depth_m
        ratios.append(ratio)
        print(f"  [{img_id}] solvePnP depth={solvepnp_depth_m*1000:.1f}mm  "
              f"ZED depth={zed_depth_mm:.1f}mm  ratio(zed/solvepnp)={ratio:.4f}",
              flush=True)

    if not ratios:
        print("[fatal] no frames yielded both a board detection and valid "
              "depth at the projected origin -- cannot check board scale "
              "this way", flush=True)
        return

    ratios = np.array(ratios)
    print(f"\n[summary] {len(ratios)} samples (of {n_checked} board "
          f"detections checked)", flush=True)
    print(f"  ratio median={np.median(ratios):.4f} mean={ratios.mean():.4f} "
          f"std={ratios.std():.4f} min={ratios.min():.4f} max={ratios.max():.4f}",
          flush=True)
    implied_square_mm = args.square_size_m * 1000.0 * np.median(ratios)
    print(f"\n[implied] if this ratio is real (not noise), the true printed "
          f"square size is {implied_square_mm:.2f}mm, vs the "
          f"{args.square_size_m*1000:.1f}mm assumed.", flush=True)
    print("[note] ratio ~1.00 (within ZED depth noise, a few percent) means "
          "the board scale assumption is fine and this is not the bug. "
          "A ratio consistently and substantially off 1.00 across samples "
          "means it is.", flush=True)


if __name__ == "__main__":
    main()
