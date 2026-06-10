"""
End-effector + wrench-line projection into the camera image.

This is the calibrated successor to the hand-fractioned auto_seed.py and the
uncalibrated force_overlay.py. It does two things, both gated on real
calibration in calibration.yaml:

  1. EE projection (unblocks writeup step 2):
       project the current_pose point into the image -> a geometric SAM seed
       at each event, replacing the vision-only heuristic in auto_seed.py.

  2. Wrench-line projection (the research direction):
       from the wrist wrench (Bicchi 1990): the force line of action is
           r0 = (f x tau)/|f|^2 ,  direction f_hat = f/|f|
       transform that ray base->camera, project it, and draw the contact ray.
       The point where it meets the contact-receiving object is the cup.

Until calibration.yaml has the relevant blocks marked `filled: true`, the
script runs in DRY mode: it prints the math it WOULD do and the base-frame
quantities, but draws no projected geometry (so we never ship fake pixels).

Go/no-go pre-test for the paper: once extrinsics arrive, run this on the
press frame and check whether the projected wrench ray lands on the cup mask.

Usage:
    .venv_analysis/bin/python Code/project_ee.py --trial Data/lfdws_t001/lfdws_t001
"""
import argparse
import ast
import os
import sys

import cv2
import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

IMG_DIR_NAME = "zed_zed_node_rgb_color_rect_image_compressed"
CALIB_DEFAULT = "calibration.yaml"

POSE_TS = "NS_1.franka_robot_state_broadcaster.current_pose.timestamp"
PX = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.x"
PY = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.y"
PZ = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.z"
QX = "NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation.x"
QY = "NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation.y"
QZ = "NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation.z"
QW = "NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation.w"
FX = "bota_post.wrench_body_compensated.wrench.force.x"
FY = "bota_post.wrench_body_compensated.wrench.force.y"
FZ = "bota_post.wrench_body_compensated.wrench.force.z"
TXc = "bota_post.wrench_body_compensated.wrench.torque.x"
TYc = "bota_post.wrench_body_compensated.wrench.torque.y"
TZc = "bota_post.wrench_body_compensated.wrench.torque.z"
GRIP = "NS_1.franka_gripper.joint_states.position"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"


# ----------------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------------
def load_calibration(path):
    if yaml is None:
        print("[fatal] pyyaml not installed in this venv "
              "(pip install pyyaml)", flush=True)
        sys.exit(1)
    if not os.path.exists(path):
        print(f"[fatal] {path} not found", flush=True)
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def quat_to_R(x, y, z, w):
    n = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def project_points(pts_base, K, T_base_cam, dist):
    """pts_base: (N,3) in base frame -> (N,2) pixels. Returns also z_cam."""
    T_cam_base = np.linalg.inv(np.array(T_base_cam, dtype=float))
    P = np.hstack([pts_base, np.ones((len(pts_base), 1))])
    cam = (T_cam_base @ P.T).T[:, :3]   # points in camera frame
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    uv, _ = cv2.projectPoints(cam.reshape(-1, 1, 3), rvec, tvec,
                              np.array(K, dtype=float),
                              np.array(dist, dtype=float).reshape(-1, 1))
    return uv.reshape(-1, 2), cam[:, 2]


# ----------------------------------------------------------------------------
# events (same logic as analyze_demo / auto_seed)
# ----------------------------------------------------------------------------
def parse_gw(c):
    try:
        return float(np.sum(ast.literal_eval(c)))
    except Exception:
        return float("nan")


def detect_events(df):
    t = pd.to_datetime(df[POSE_TS])
    t_rel = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    w = df[GRIP].apply(parse_gw).to_numpy()
    fm = np.sqrt(df[FX].astype(float)**2 + df[FY].astype(float)**2 +
                 df[FZ].astype(float)**2).to_numpy()
    w_open, w_closed = float(np.nanmax(w)), float(np.nanmin(w))
    thr = w_closed + 0.5*(w_open - w_closed)
    closed = w < thr
    cd = np.where((~closed[:-1]) & (closed[1:]))[0] + 1
    cu = np.where((closed[:-1]) & (~closed[1:]))[0] + 1
    out = {}
    if len(cd):
        i = int(cd[0]);  out["grasp"] = i
    if len(cu):
        i = int(cu[-1]); out["release"] = i
    fm_adj = np.where(closed, fm - np.median(fm[:len(fm)//10]), -np.inf)
    out["press"] = int(np.argmax(fm_adj))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--calib", default=CALIB_DEFAULT)
    ap.add_argument("--out", default="figures/identify/projection.png")
    ap.add_argument("--ray_len_m", type=float, default=0.12,
                    help="length of the drawn contact ray along -force (metres)")
    args = ap.parse_args()

    calib = load_calibration(args.calib)
    intr_ok = calib.get("camera_intrinsics", {}).get("filled", False)
    extr_ok = calib.get("base_to_camera", {}).get("filled", False)
    bota_ok = calib.get("bota_to_base", {}).get("filled", False)
    dry = not (intr_ok and extr_ok)
    print(f"[calib] intrinsics={intr_ok} extrinsics={extr_ok} bota_mount={bota_ok}",
          flush=True)
    if dry:
        print("[calib] DRY MODE — intrinsics/extrinsics not yet provided; "
              "reporting base-frame geometry only, drawing nothing.", flush=True)

    csv_path = next((os.path.join(args.trial, f) for f in os.listdir(args.trial)
                     if f.endswith(".csv") and not f.startswith(".")), None)
    img_dir = os.path.join(args.trial, IMG_DIR_NAME)
    df = pd.read_csv(csv_path)
    events = detect_events(df)
    print(f"[load] {csv_path} ({len(df)} rows)", flush=True)

    ee_cfg = calib.get("end_effector", {})
    is_tcp = ee_cfg.get("current_pose_is", "tcp") == "tcp"
    flange_to_tcp_z = float(ee_cfg.get("flange_to_tcp_z", 0.1034))

    panels = []
    for name, idx in events.items():
        r = df.iloc[idx]
        ee = np.array([r[PX], r[PY], r[PZ]], dtype=float)
        R_be = quat_to_R(r[QX], r[QY], r[QZ], r[QW])  # base<-EE rotation
        # if current_pose is the flange, step +z by flange_to_tcp_z to reach TCP
        if not is_tcp:
            ee = ee + R_be @ np.array([0, 0, flange_to_tcp_z])
        img_id = str(r[IMG])
        print(f"\n[event] {name}  row={idx}  img={img_id}", flush=True)
        print(f"  EE/TCP (base) = {ee.round(4).tolist()}", flush=True)

        # wrench (bota frame)
        f_b = np.array([r[FX], r[FY], r[FZ]], dtype=float)
        tau_b = np.array([r[TXc], r[TYc], r[TZc]], dtype=float)
        fmag = float(np.linalg.norm(f_b))
        print(f"  |F| = {fmag:.2f} N   f_bota = {f_b.round(3).tolist()}", flush=True)
        if fmag > 1e-3:
            f_hat_b = f_b / fmag
            r0_b = np.cross(f_b, tau_b) / (fmag**2)   # Bicchi closest point
            print(f"  Bicchi r0 (bota) = {r0_b.round(4).tolist()}  "
                  f"f_hat = {f_hat_b.round(3).tolist()}", flush=True)
        else:
            f_hat_b = np.zeros(3); r0_b = np.zeros(3)

        bgr = cv2.imread(os.path.join(img_dir, f"{img_id}.png"))
        if bgr is None:
            print(f"  [skip] missing frame", flush=True); continue

        if not dry:
            K = calib["camera_intrinsics"]["K"]
            dist = calib["camera_intrinsics"]["dist"]
            T_bc = calib["base_to_camera"]["T"]
            # EE point
            uv_ee, z = project_points(ee.reshape(1, 3), K, T_bc, dist)
            u, v = uv_ee[0]
            print(f"  EE -> pixel ({u:.0f},{v:.0f})  z_cam={z[0]:.3f}m", flush=True)
            ov = bgr.copy()
            cv2.circle(ov, (int(u), int(v)), 9, (0, 0, 255), -1)
            cv2.circle(ov, (int(u), int(v)), 9, (255, 255, 255), 2)

            # wrench ray (needs bota mount)
            if bota_ok and fmag > 1e-3:
                T_bb = np.array(calib["bota_to_base"]["T"], dtype=float)
                R_bb = T_bb[:3, :3]
                f_hat_base = R_bb @ f_hat_b
                # draw from EE along -force (into the contact)
                tip = ee + (-f_hat_base) * args.ray_len_m
                uv2, _ = project_points(np.vstack([ee, tip]), K, T_bc, dist)
                p0 = (int(uv2[0][0]), int(uv2[0][1]))
                p1 = (int(uv2[1][0]), int(uv2[1][1]))
                cv2.arrowedLine(ov, p0, p1, (0, 0, 255), 4, tipLength=0.25)
                print(f"  wrench ray -> {p0} to {p1}", flush=True)
            else:
                cv2.putText(ov, "wrench ray: needs bota_to_base", (10, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.putText(ov, f"{name}  EE proj  |F|={fmag:.1f}N", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            panels.append(ov)
        else:
            ov = bgr.copy()
            cv2.putText(ov, f"{name}  DRY (awaiting calibration)", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            panels.append(ov)

    if panels:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        th = 480
        resized = [cv2.resize(p, (int(p.shape[1]*th/p.shape[0]), th)) for p in panels]
        cv2.imwrite(args.out, np.hstack(resized))
        print(f"\n[write] {args.out}", flush=True)
    status = "DRY (awaiting calibration)" if dry else "LIVE"
    print(f"[done] mode={status}", flush=True)


if __name__ == "__main__":
    main()
