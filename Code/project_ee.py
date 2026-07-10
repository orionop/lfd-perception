"""
End-effector + wrench-line projection into the camera image.

EYE-IN-HAND rig model (per the lab's 2026-07-09 email + CAD, see
calibration.yaml header for the full derivation): the ZED Mini is mounted on
a bracket bolted to the gripper, not fixed in the world. `current_pose` is
the pose of the BOTA SENSONE ORIGIN in the base frame (not the TCP), so the
camera's pose in the base frame is a PER-FRAME quantity:

    T_base_camera(t) = T_base_bota(t) @ T_bota_camera

where T_base_bota(t) comes straight from current_pose's position+quaternion
at that row, and T_bota_camera is the one FIXED unknown -- rigid bracket
geometry from calibration.yaml's bota_to_camera block.

This dissolves the old "bota_frame -> base mount transform" ask: the wrench
(measured in the bota frame) rotates into the base frame using the SAME
current_pose rotation used for the camera, no separate mount calibration
needed.

Two things this script does, both gated on calibration.yaml's
camera_intrinsics + bota_to_camera being filled:

  1. EE projection (unblocks writeup step 2):
       project the bota-origin point (current_pose) into the image -> a
       geometric SAM seed at each event, replacing the vision-only heuristic
       in auto_seed.py.

  2. Wrench-line projection (the research direction):
       from the wrist wrench (Bicchi 1990): the force line of action is
           r0 = (f x tau)/|f|^2 ,  direction f_hat = f/|f|
       (both in the bota frame) -- rotate+translate into the base frame via
       T_base_bota(t), then project into the image and draw the contact ray.

Until calibration.yaml has the relevant blocks marked `filled: true`, the
script runs in DRY mode: it prints the math it WOULD do and the base-frame
quantities, but draws no projected geometry (so we never ship fake pixels).
NOTE: as of the 2026-07-10 CAD extraction, bota_to_camera is filled with a
documented PRELIMINARY estimate (still filled: false) -- flip that flag only
once the value is trusted (see calibration.yaml's caveats: left/right lens
sign, zero lens-depth-offset assumption).

Go/no-go pre-test for the paper: once bota_to_camera is trusted, run this on
the press frame and check whether the projected wrench ray lands on the
contact-receiving object's mask.

Usage:
    .venv_analysis/bin/python Code/project_ee.py --trial Data/lfdws_t002_new
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


def pose_to_T(px, py, pz, qx, qy, qz, qw):
    """current_pose row -> 4x4 T_base_bota(t)."""
    T = np.eye(4)
    T[:3, :3] = quat_to_R(qx, qy, qz, qw)
    T[:3, 3] = [px, py, pz]
    return T


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
    """Same force-only / gripper-only fallbacks as auto_seed.py."""
    t = pd.to_datetime(df[POSE_TS])
    t_rel = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    has_force = FX in df.columns
    if has_force:
        fm = np.sqrt(df[FX].astype(float)**2 + df[FY].astype(float)**2 +
                     df[FZ].astype(float)**2).to_numpy()
        baseline = np.median(fm[:len(fm)//10])

    if GRIP not in df.columns:
        if not has_force:
            return {}
        fm_adj = fm - baseline
        return {"press": int(np.argmax(fm_adj))}

    w = df[GRIP].apply(parse_gw).to_numpy()
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
    if not has_force:
        return out
    fm_adj = np.where(closed, fm - baseline, -np.inf)
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
    bota_cam_ok = calib.get("bota_to_camera", {}).get("filled", False)
    dry = not (intr_ok and bota_cam_ok)
    print(f"[calib] intrinsics={intr_ok} bota_to_camera={bota_cam_ok}", flush=True)
    if dry:
        print("[calib] DRY MODE — camera_intrinsics/bota_to_camera not yet "
              "filled:true; reporting bota-frame geometry only, drawing "
              "nothing.", flush=True)

    csv_path = next((os.path.join(args.trial, f) for f in os.listdir(args.trial)
                     if f.endswith(".csv") and not f.startswith(".")), None)
    img_dir = os.path.join(args.trial, IMG_DIR_NAME)
    df = pd.read_csv(csv_path)
    events = detect_events(df)
    print(f"[load] {csv_path} ({len(df)} rows)", flush=True)
    if not events:
        print("[fatal] no events detectable (no gripper and no wrench topic)",
              flush=True)
        return

    ee_cfg = calib.get("end_effector", {})
    is_bota_origin = ee_cfg.get("current_pose_is", "tcp") == "bota_origin"
    bota_to_tcp = ee_cfg.get("bota_to_tcp", None)

    panels = []
    for name, idx in events.items():
        r = df.iloc[idx]
        bota_pos = np.array([r[PX], r[PY], r[PZ]], dtype=float)
        T_base_bota = pose_to_T(r[PX], r[PY], r[PZ], r[QX], r[QY], r[QZ], r[QW])
        R_base_bota = T_base_bota[:3, :3]

        # anchor point for seeding: bota origin, optionally offset to the
        # fingertip TCP once that CAD offset is known (still null as of
        # 2026-07-10 -- see calibration.yaml end_effector block)
        anchor = bota_pos.copy()
        if is_bota_origin and bota_to_tcp is not None:
            anchor = anchor + R_base_bota @ np.array(bota_to_tcp, dtype=float)

        img_id = str(r[IMG])
        print(f"\n[event] {name}  row={idx}  img={img_id}", flush=True)
        print(f"  bota origin (base) = {bota_pos.round(4).tolist()}", flush=True)
        if bota_to_tcp is None:
            print("  [note] bota_to_tcp offset not yet known -- anchor = "
                  "bota origin, not the true fingertip TCP", flush=True)

        # wrench (bota frame)
        has_force = FX in df.columns
        if has_force:
            f_b = np.array([r[FX], r[FY], r[FZ]], dtype=float)
            tau_b = np.array([r[TXc], r[TYc], r[TZc]], dtype=float)
            fmag = float(np.linalg.norm(f_b))
            print(f"  |F| = {fmag:.2f} N   f_bota = {f_b.round(3).tolist()}", flush=True)
        else:
            fmag = 0.0
            f_b = tau_b = np.zeros(3)
            print("  [note] no wrench topic in this trial -- no force/contact "
                  "geometry for this event", flush=True)

        if fmag > 1e-3:
            f_hat_b = f_b / fmag
            r0_b = np.cross(f_b, tau_b) / (fmag**2)   # Bicchi closest point (bota frame)
            # rotate+translate into base frame via the SAME current_pose
            # transform used for the camera -- this is what dissolves the
            # old separate bota_to_base mount-transform ask.
            r0_base = T_base_bota[:3, :3] @ r0_b + T_base_bota[:3, 3]
            f_hat_base = R_base_bota @ f_hat_b
            print(f"  Bicchi r0 (bota) = {r0_b.round(4).tolist()}  "
                  f"f_hat (bota) = {f_hat_b.round(3).tolist()}", flush=True)
            print(f"  Bicchi r0 (base) = {r0_base.round(4).tolist()}  "
                  f"f_hat (base) = {f_hat_base.round(3).tolist()}", flush=True)
        else:
            f_hat_base = np.zeros(3); r0_base = np.zeros(3)

        bgr = cv2.imread(os.path.join(img_dir, f"{img_id}.png"))
        if bgr is None:
            print(f"  [skip] missing frame", flush=True); continue

        if not dry:
            K = calib["camera_intrinsics"]["K"]
            dist = calib["camera_intrinsics"]["dist"]
            T_bota_cam = np.array(calib["bota_to_camera"]["T"], dtype=float)
            T_base_cam = T_base_bota @ T_bota_cam   # PER-FRAME (eye-in-hand)

            uv_ee, z = project_points(anchor.reshape(1, 3), K, T_base_cam, dist)
            u, v = uv_ee[0]
            print(f"  anchor -> pixel ({u:.0f},{v:.0f})  z_cam={z[0]:.3f}m", flush=True)
            ov = bgr.copy()
            cv2.circle(ov, (int(u), int(v)), 9, (0, 0, 255), -1)
            cv2.circle(ov, (int(u), int(v)), 9, (255, 255, 255), 2)

            if fmag > 1e-3:
                tip = r0_base + (-f_hat_base) * args.ray_len_m
                uv2, _ = project_points(np.vstack([r0_base, tip]), K, T_base_cam, dist)
                p0 = (int(uv2[0][0]), int(uv2[0][1]))
                p1 = (int(uv2[1][0]), int(uv2[1][1]))
                cv2.arrowedLine(ov, p0, p1, (0, 0, 255), 4, tipLength=0.25)
                print(f"  wrench ray -> {p0} to {p1}", flush=True)
            else:
                cv2.putText(ov, "wrench ray: no force this event", (10, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.putText(ov, f"{name}  anchor proj  |F|={fmag:.1f}N", (10, 28),
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
