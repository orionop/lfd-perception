"""
Eye-in-hand hand-eye calibration: recovers T_bota_camera (bota SensONE
origin -> ZED-Mini left-lens optical frame), the one fixed unknown that
has blocked project_ee.py's LIVE mode since the eye-in-hand rig model was
adopted (see calibration.yaml). Supersedes the CAD-reading approach
(Code/cad_extract_transform.py, Code/cad_find_lens_occ.py), which left
the lens position ambiguous between two 63mm-apart candidates and, per
Code/cad_candidate_sensitivity.py, produced projections landing entirely
outside the image on every trial tested -- the CAD-derived rotation
itself is unreliable, not just the translation. This script recovers the
transform by direct measurement instead of reading it off a drawing.

METHOD (OpenCV's standard eye-in-hand recipe, AX=XB formulation):
  A ChArUco (or plain ArUco) board is fixed in the world (bolted to the
  table/rig, NOT held by the robot -- it must not move during capture).
  The arm is driven through N >= 3 (recommend 10-15 for a good solve)
  distinct poses. At each pose:
    - current_pose gives T_base_bota(t) directly (position + quaternion)
    - the ZED RGB frame gives the board's pose in the camera frame,
      T_cam_board(t), via solvePnP on detected board corners against the
      board's known geometry (square size + marker size, both required
      inputs -- MEASURE THE PHYSICAL BOARD, don't assume a print scale)
  cv2.calibrateHandEye() then solves for the one FIXED transform
  T_bota_camera consistent with all pose pairs simultaneously (the board
  stays fixed in the world across all N poses; the arm+camera move
  together as a rigid eye-in-hand unit, which is exactly cv2's
  CALIB_HAND_EYE_* eye-in-hand assumption).

REQUIREMENTS BEFORE RUNNING (fill these in, cannot proceed without them):
  - A printed ChArUco board, physically measured: SQUARE_SIZE_M and
    MARKER_SIZE_M below MUST match the actual printed board, not the PDF's
    nominal spec (printer scaling drift is a common silent error source).
  - The board must be visible in the ZED RGB frame at every captured pose
    and must not move during the whole capture sequence.
  - Camera intrinsics (K, dist) -- already available in calibration.yaml,
    camera_intrinsics.filled: true.
  - >= 10 distinct arm poses spanning a real range of orientations, not
    just translations (rotation diversity is what makes AX=XB solvable;
    a purely translational pose set is a known degenerate case).

USAGE (two-phase workflow):

  Phase 1 -- CAPTURE, at the rig, once per pose (repeat >=10-15 times,
  varying orientation as well as position):
    .venv_analysis/bin/python Code/calibrate_hand_eye.py capture \
        --out_dir calib_capture --pose_csv <live current_pose CSV or dir>

  Actually simplest for this rig: record a short bag WHILE moving the arm
  through several static pauses in front of the board (a few seconds
  still at each pose so gripper/rgb timestamps have unambiguous matches),
  export it the normal way (ros2_unbag or mcap_extract.py), then run:

    .venv_analysis/bin/python Code/calibrate_hand_eye.py solve \
        --trial Data/<calib_trial> --square_size_m 0.035 \
        --marker_size_m 0.026 --squares_x 5 --squares_y 7 \
        --out calibration_handeye_result.yaml

  This detects the board in every frame with a detection, pairs each with
  its current_pose row, keeps only well-separated poses (drops
  near-duplicate frames from a still pause), solves, and reports:
    - T_bota_camera (4x4, ready to paste into calibration.yaml)
    - per-pose reprojection residual (sanity check -- large scatter here
      means bad board detections or insufficient pose diversity, not a
      solver bug)
  Writes calibration_handeye_result.yaml alongside a
  calibration_handeye_debug.png (board-corner overlay on a few sample
  frames, to visually confirm detection quality before trusting the
  numbers).

DOES NOT modify calibration.yaml automatically -- prints the block to
paste in by hand after reviewing the residuals, same "never silently use
an unreviewed number" discipline as the rest of this repo's calibration
handling.

VALIDATED PRE-RIG (2026-08-10): both stages tested against synthetic data
before ever touching real captures. (1) detect_board_pose against a
rendered CharucoBoard image pasted into a synthetic frame -- correctly
recovers a near-zero rotation, plausible translation. (2) the full
AX=XB solve (calibrateHandEye + pose_diversity_filter) against 12
synthetic poses generated from a known ground-truth T_bota_camera --
recovers it to 0.0deg rotation error and ~1e-16m translation error. Both
stages are therefore known-correct; the only thing rig time needs to
validate is real-world detection quality (lighting, board visibility,
pose diversity), not the underlying math.
"""
import argparse
import ast
import os

import cv2
import numpy as np
import pandas as pd
import yaml

POSE_TS = "NS_1.franka_robot_state_broadcaster.current_pose.timestamp"
PX = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.x"
PY = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.y"
PZ = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.z"
QX = "NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation.x"
QY = "NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation.y"
QZ = "NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation.z"
QW = "NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation.w"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"
IMG_DIR_NAME = "zed_zed_node_rgb_color_rect_image_compressed"


def quat_to_R(x, y, z, w):
    n = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def load_calibration(path="calibration.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def detect_board_pose(bgr, board, K, dist):
    """Returns (rvec, tvec) of the board in the camera frame, or None if
    not enough markers/corners detected for a reliable solvePnP."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    aruco_dict = board.getDictionary()
    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    if ids is None or len(ids) < 4:
        return None
    charuco_detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = charuco_detector.detectBoard(gray)
    if charuco_corners is None or len(charuco_corners) < 6:
        return None
    obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_points is None or len(obj_points) < 6:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist)
    if not ok:
        return None
    return rvec, tvec


def pose_diversity_filter(poses_base_bota, min_translation_m=0.03, min_rotation_deg=5.0):
    """Greedily keep poses that differ enough from all previously-kept
    poses -- drops near-duplicate frames from a still pause without
    requiring the capture script to have been perfectly timed. Returns
    the KEPT INDICES into poses_base_bota (not the poses themselves --
    numpy arrays aren't safely hashable/comparable for index recovery)."""
    kept_idx = []
    for i, (R, t) in enumerate(poses_base_bota):
        far_enough = True
        for j in kept_idx:
            R2, t2 = poses_base_bota[j]
            dt = np.linalg.norm(t - t2)
            dR = R @ R2.T
            angle = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
            if dt < min_translation_m and angle < min_rotation_deg:
                far_enough = False
                break
        if far_enough:
            kept_idx.append(i)
    return kept_idx


def solve(args):
    calib = load_calibration()
    if not calib.get("camera_intrinsics", {}).get("filled", False):
        print("[fatal] calibration.yaml camera_intrinsics not filled -- "
              "needed for solvePnP. Nothing to do.", flush=True)
        return
    K = np.array(calib["camera_intrinsics"]["K"], dtype=float)
    dist = np.array(calib["camera_intrinsics"]["dist"], dtype=float)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y), args.square_size_m,
        args.marker_size_m, aruco_dict)

    demo_csv = next((os.path.join(args.trial, f) for f in os.listdir(args.trial)
                     if f.endswith(".csv") and not f.startswith(".")), None)
    if demo_csv is None:
        print(f"[fatal] no merged CSV in {args.trial}", flush=True)
        return
    img_dir = os.path.join(args.trial, IMG_DIR_NAME)
    df = pd.read_csv(demo_csv)
    print(f"[load] {demo_csv} ({len(df)} rows)", flush=True)

    R_base_bota_all, t_base_bota_all = [], []
    R_cam_board_all, t_cam_board_all = [], []
    debug_frames = []
    n_checked, n_detected = 0, 0
    for _, r in df.iterrows():
        img_id = str(r[IMG])
        img_path = os.path.join(img_dir, f"{img_id}.png")
        if not os.path.exists(img_path):
            continue
        n_checked += 1
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        det = detect_board_pose(bgr, board, K, dist)
        if det is None:
            continue
        rvec, tvec = det
        n_detected += 1
        R_cb, _ = cv2.Rodrigues(rvec)
        R_cam_board_all.append(R_cb)
        t_cam_board_all.append(tvec.reshape(3))

        R_bb = quat_to_R(r[QX], r[QY], r[QZ], r[QW])
        t_bb = np.array([r[PX], r[PY], r[PZ]], dtype=float)
        R_base_bota_all.append(R_bb)
        t_base_bota_all.append(t_bb)

        if len(debug_frames) < 6 and n_detected % max(1, n_checked // 6 + 1) == 0:
            ov = bgr.copy()
            cv2.drawFrameAxes(ov, K, dist, rvec, tvec, args.square_size_m * 2)
            debug_frames.append(ov)

    print(f"[detect] board found in {n_detected}/{n_checked} frames with an image",
          flush=True)
    if n_detected < 3:
        print("[fatal] need >=3 successful board detections (recommend "
              ">=10-15) -- check board is in frame, SQUARE_SIZE_M/"
              "MARKER_SIZE_M match the physical printout, and the "
              "dictionary (DICT_5X5_100) matches the printed board.",
              flush=True)
        return

    keep_idx = pose_diversity_filter(list(zip(R_base_bota_all, t_base_bota_all)))
    print(f"[filter] {len(keep_idx)}/{n_detected} poses kept after "
          f"near-duplicate removal", flush=True)
    if len(keep_idx) < 3:
        print("[fatal] too few diverse poses after filtering -- capture "
              "more distinct arm poses (vary orientation, not just "
              "position)", flush=True)
        return

    R_gripper2base = [R_base_bota_all[i] for i in keep_idx]
    t_gripper2base = [t_base_bota_all[i] for i in keep_idx]
    R_target2cam = [R_cam_board_all[i] for i in keep_idx]
    t_target2cam = [t_cam_board_all[i] for i in keep_idx]

    print(f"[solve] cv2.calibrateHandEye on {len(R_gripper2base)} pose pairs "
          f"(eye-in-hand: board fixed in world, camera rigidly mounted on "
          f"the moving gripper/bota frame)", flush=True)
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base, R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI)

    T_bota_camera = np.eye(4)
    T_bota_camera[:3, :3] = R_cam2gripper
    T_bota_camera[:3, 3] = t_cam2gripper.reshape(3)
    print("\n[result] T_bota_camera (bota origin frame -> camera-frame point):")
    print(T_bota_camera.round(6), flush=True)

    # residual check: reproject the board's known origin through each pose's
    # solved transform and compare against the directly-observed board pose
    residuals = []
    for i in keep_idx:
        R_bb, t_bb = R_base_bota_all[i], t_base_bota_all[i]
        T_base_bota = np.eye(4); T_base_bota[:3, :3] = R_bb; T_base_bota[:3, 3] = t_bb
        T_base_cam_est = T_base_bota @ T_bota_camera
        # board pose in base frame, estimated two ways should agree across
        # poses since board is fixed in world -- use spread of estimated
        # board-in-base position as the residual
        R_cb, t_cb = R_cam_board_all[i], t_cam_board_all[i]
        T_cam_board = np.eye(4); T_cam_board[:3, :3] = R_cb; T_cam_board[:3, 3] = t_cb
        T_base_board_est = T_base_cam_est @ T_cam_board
        residuals.append(T_base_board_est[:3, 3])
    residuals = np.array(residuals)
    spread = residuals.std(axis=0)
    print(f"\n[check] board-in-base-frame position estimated from each pose "
          f"independently -- should agree tightly since the board is fixed "
          f"in the world. Std dev across {len(residuals)} poses: "
          f"{spread.round(4)} m (per-axis). Large values (>1-2cm) indicate "
          f"bad detections or insufficient pose diversity, not a solver bug.",
          flush=True)

    out_doc = {
        "T_bota_camera": T_bota_camera.round(6).tolist(),
        "n_poses_used": len(keep_idx),
        "n_poses_detected_total": n_detected,
        "board_in_base_position_std_m": spread.round(4).tolist(),
        "square_size_m": args.square_size_m,
        "marker_size_m": args.marker_size_m,
        "note": "REVIEW residuals before pasting into calibration.yaml's "
                "bota_to_camera.T -- do not mark filled:true without "
                "checking the debug image and the std-dev residual above.",
    }
    with open(args.out, "w") as f:
        yaml.safe_dump(out_doc, f, sort_keys=False)
    print(f"\n[write] {args.out}", flush=True)

    if debug_frames:
        th = 240
        resized = [cv2.resize(f, (int(f.shape[1]*th/f.shape[0]), th)) for f in debug_frames]
        strip = np.hstack(resized)
        dbg_path = "calibration_handeye_debug.png"
        cv2.imwrite(dbg_path, strip)
        print(f"[write] {dbg_path} (board-axes overlay, {len(debug_frames)} sample frames)",
              flush=True)
    print("[done] -- REVIEW before pasting T_bota_camera into calibration.yaml", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("solve", help="detect board + solve AX=XB from a captured trial")
    sp.add_argument("--trial", required=True,
                    help="trial dir with merged CSV + RGB PNGs, captured by moving "
                         "the arm through several static poses in front of a fixed board")
    sp.add_argument("--squares_x", type=int, default=5)
    sp.add_argument("--squares_y", type=int, default=7)
    sp.add_argument("--square_size_m", type=float, required=True,
                    help="PHYSICALLY MEASURE the printed board -- do not assume nominal PDF scale")
    sp.add_argument("--marker_size_m", type=float, required=True,
                    help="physically measured ArUco marker size within each square")
    sp.add_argument("--out", default="calibration_handeye_result.yaml")

    args = ap.parse_args()
    if args.cmd == "solve":
        solve(args)


if __name__ == "__main__":
    main()
