"""
Independently re-derive camera_intrinsics (K, dist) from the same ChArUco
recording, and compare against calibration.yaml's lab-provided values, which
have never been re-verified against real data in this repo -- "source:
lab-provided intrinsics.md, 2026-07-09" is a transcribed number, not
something this pipeline has ever cross-checked.

METHOD
------
Standard OpenCV camera calibration: accumulate ChArUco object/image point
correspondences across many diverse, well-detected, near-static frames from
charuco_calib_002, then cv2.calibrateCamera(). This uses the SAME detection
pass and pose-diversity filter as calibrate_hand_eye.py (import, not a
reimplementation) so the frame set is identical to what the hand-eye solve
actually used -- if intrinsics are the problem, it should show up on exactly
the data the hand-eye calibration relied on.

This does NOT write calibration.yaml. Prints for manual comparison only.

Usage:
    .venv_analysis/bin/python Code/verify_camera_intrinsics.py \
        --trial Data/charuco_calib_002 --square_size_m 0.035 --marker_size_m 0.026 \
        --squares_x 5 --squares_y 7 --max_speed_mps 0.001
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_hand_eye import (IMG, IMG_DIR_NAME, PX, PY, PZ, POSE_TS,
                                load_calibration, pose_diversity_filter,
                                quat_to_R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--squares_x", type=int, default=5)
    ap.add_argument("--squares_y", type=int, default=7)
    ap.add_argument("--square_size_m", type=float, required=True)
    ap.add_argument("--marker_size_m", type=float, required=True)
    ap.add_argument("--max_speed_mps", type=float, default=None)
    args = ap.parse_args()

    calib = load_calibration()
    lab_K = np.array(calib["camera_intrinsics"]["K"], dtype=float)
    lab_dist = np.array(calib["camera_intrinsics"]["dist"], dtype=float)
    print(f"[lab] K=\n{lab_K}\n[lab] dist={lab_dist}", flush=True)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y), args.square_size_m,
        args.marker_size_m, aruco_dict)
    charuco_detector = cv2.aruco.CharucoDetector(board)

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

    # First pass, matching calibrate_hand_eye.py exactly: find every
    # well-detected frame's pose, so pose_diversity_filter picks the same
    # kind of well-spread subset the hand-eye solve used.
    all_obj, all_img, all_poses, all_shape = [], [], [], None
    seen = set()
    for i, r in df.iterrows():
        if speed is not None:
            s = speed[i]
            if not np.isfinite(s) or s >= args.max_speed_mps:
                continue
        img_id = str(r[IMG])
        if img_id in seen:
            continue
        seen.add(img_id)
        img_path = os.path.join(img_dir, f"{img_id}.png")
        if not os.path.exists(img_path):
            continue
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if all_shape is None:
            all_shape = gray.shape[::-1]  # (w, h)
        charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
        if charuco_corners is None or len(charuco_corners) < 6:
            continue
        obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
        if obj_points is None or len(obj_points) < 6:
            continue
        R_bb = quat_to_R(r["NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation.x"],
                         r["NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation.y"],
                         r["NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation.z"],
                         r["NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation.w"])
        t_bb = np.array([r[PX], r[PY], r[PZ]], dtype=float)
        all_obj.append(obj_points)
        all_img.append(img_points)
        all_poses.append((R_bb, t_bb))

    print(f"[detect] {len(all_obj)} usable frames with >=6 charuco corners",
          flush=True)
    keep_idx = pose_diversity_filter(all_poses)
    print(f"[filter] {len(keep_idx)}/{len(all_obj)} kept after near-duplicate "
          f"removal (same diversity filter as calibrate_hand_eye.py)",
          flush=True)
    if len(keep_idx) < 8:
        print("[fatal] too few diverse frames for a meaningful intrinsics "
              "estimate", flush=True)
        return

    obj_pts = [all_obj[i] for i in keep_idx]
    img_pts = [all_img[i] for i in keep_idx]

    print(f"[calibrate] cv2.calibrateCamera on {len(obj_pts)} frames ...",
          flush=True)
    rms, K_new, dist_new, _, _ = cv2.calibrateCamera(
        obj_pts, img_pts, all_shape, None, None)

    print(f"\n[result] reprojection RMS error: {rms:.4f} px", flush=True)
    print(f"[result] fresh K=\n{K_new}", flush=True)
    print(f"[result] fresh dist={dist_new.ravel()}", flush=True)

    dK = K_new - lab_K
    print(f"\n[compare] K difference (fresh - lab):\n{dK.round(3)}", flush=True)
    print(f"[compare] fx: lab={lab_K[0,0]:.2f} fresh={K_new[0,0]:.2f} "
          f"({100*(K_new[0,0]-lab_K[0,0])/lab_K[0,0]:+.2f}%)", flush=True)
    print(f"[compare] fy: lab={lab_K[1,1]:.2f} fresh={K_new[1,1]:.2f} "
          f"({100*(K_new[1,1]-lab_K[1,1])/lab_K[1,1]:+.2f}%)", flush=True)
    print(f"[compare] cx: lab={lab_K[0,2]:.2f} fresh={K_new[0,2]:.2f}", flush=True)
    print(f"[compare] cy: lab={lab_K[1,2]:.2f} fresh={K_new[1,2]:.2f}", flush=True)
    print(f"[compare] lab assumed dist=0; fresh dist magnitude="
          f"{np.abs(dist_new.ravel()).max():.5f}", flush=True)
    print("\n[note] this is a standalone check, not a calibration.yaml "
          "write -- a large fx/fy or dist difference here would implicate "
          "camera_intrinsics as (part of) the wrench-ray gap; a small one "
          "rules it out the same way the previous five checks did.",
          flush=True)


if __name__ == "__main__":
    main()
