"""
CAD candidate A/B bota_to_camera sensitivity analysis.

calibration.yaml's bota_to_camera stays filled: false because the CAD
extraction left one unresolved ambiguity: which of two candidate lens
positions (A or B, ~63mm apart along the housing's local X, i.e. the ZED
Mini's own stereo baseline direction) is the true left-lens position. Mark's
CAD person will eventually resolve this, but that has no ETA. Rather than
producing zero result until then, this script runs the SAME wrench-ray
projection math as Code/project_ee.py with BOTH candidates on every trial
that has a real force event AND an already-propagated ground-truth
contact-receiver mask, and reports the projected ray's distance to that
mask's bbox centroid for each candidate. This bounds the expected error
range under the unresolved calibration ambiguity, rather than reporting
nothing.

Does not modify calibration.yaml or project_ee.py -- standalone, reads the
shared rotation block from calibration.yaml and swaps in each candidate's
translation.

Trials used (real force + real propagated receiver mask):
  - lfdws_t001        press -> cup (contact_receiver)
  - lfdws_t001_depth  press -> plate (contact_receiver)
  - lfdws_t001_labexport press -> latch/handle (contact_receiver)
(lfdws_t002_labexport's press event has no discrete receiver -- cube
pressed against the table, not a bounded object -- excluded, see
figures/t002labexport's build_sidecar_multi.py run notes.)

Output: figures/cad_candidate_sensitivity.png (one row per trial, columns
= candidate A / candidate B, ray drawn + receiver bbox outline + pixel
distance) and printed summary table.

Run inside .venv_analysis:
    .venv_analysis/bin/python Code/cad_candidate_sensitivity.py
"""
import ast
import csv
import os

import cv2
import numpy as np

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from event_utils import gripper_moved
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
FX = "bota_post.wrench_body_compensated.wrench.force.x"
FY = "bota_post.wrench_body_compensated.wrench.force.y"
FZ = "bota_post.wrench_body_compensated.wrench.force.z"
TXc = "bota_post.wrench_body_compensated.wrench.torque.x"
TYc = "bota_post.wrench_body_compensated.wrench.torque.y"
TZc = "bota_post.wrench_body_compensated.wrench.torque.z"
GRIP = "NS_1.franka_gripper.joint_states.position"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"

CANDIDATES = {
    "A": [0.111309, -0.143553, 0.025888],
    "B": [0.155857, -0.099006, 0.025888],
}
# shared rotation block from calibration.yaml's bota_to_camera.T (only the
# translation column differs between candidates)
R_BOTA_CAM = np.array([
    [-0.7071, -0.5,     -0.5],
    [-0.7071,  0.5,      0.5],
    [ 0.0,     0.7071,  -0.7071],
])

TRIALS = [
    {"name": "lfdws_t001", "csv": "Data/lfdws_t001/lfdws_t001/lfdws_t001_0.csv",
     "img_dir": "Data/lfdws_t001/lfdws_t001/zed_zed_node_rgb_color_rect_image_compressed",
     "sidecar": "figures/identify/objects_summary.csv", "receiver_role": "contact_receiver"},
    {"name": "lfdws_t001_depth", "csv": "Data/lfdws_t001_depth/lfdws_t001_depth_0.csv",
     "img_dir": "Data/lfdws_t001_depth/zed_zed_node_rgb_color_rect_image_compressed",
     "sidecar": "figures/identify_depth/objects_summary.csv", "receiver_role": "contact_receiver"},
    {"name": "lfdws_t001_labexport", "csv": "Data/lfdws_t001_labexport/lfdws_t001/lfdws_t001.csv",
     "img_dir": "Data/lfdws_t001_labexport/lfdws_t001/zed_zed_node_rgb_color_rect_image_compressed",
     "sidecar": "figures/t001labexport/identify/objects_summary.csv", "receiver_role": "contact_receiver"},
]

OUT = "figures/cad_candidate_sensitivity.png"


def parse_gw(c):
    try:
        return float(np.sum(ast.literal_eval(c)))
    except Exception:
        return float("nan")


def detect_press_event(df):
    """Same force-only/gripper-only fallback logic as project_ee.py."""
    has_force = FX in df.columns
    if not has_force:
        return None
    fm = np.sqrt(df[FX].astype(float)**2 + df[FY].astype(float)**2 +
                 df[FZ].astype(float)**2).to_numpy()
    baseline = np.median(fm[:len(fm)//10])
    if GRIP not in df.columns:
        return int(np.argmax(fm - baseline))
    w = df[GRIP].apply(parse_gw).to_numpy()
    w_open, w_closed = float(np.nanmax(w)), float(np.nanmin(w))
    thr = w_closed + 0.5*(w_open - w_closed)
    # Guard: a gripper that never actuated puts this midpoint INSIDE the
    # sensor's own noise band and manufactures grasp/release out of nothing.
    # Measured on lfdws_t001_labexport: width spans 6.6e-7 m of pure noise yet
    # the unguarded rule reported a grasp at 0.06 s and a release at 7.66 s,
    # which then displaced the contact event from the true 11.15 N peak at
    # 5.08 s to 3.34 s. See Code/event_utils.py.
    closed = (w < thr) if gripper_moved(w) else np.zeros(len(w), dtype=bool)
    # With no real grasp cycle there is no held window to restrict to, so
    # search the whole recording rather than an all -inf array whose argmax
    # would silently return row 0.
    fm_adj = (np.where(closed, fm - baseline, -np.inf)
              if closed.any() else fm - baseline)
    return int(np.argmax(fm_adj))


def quat_to_R(x, y, z, w):
    n = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def pose_to_T(px, py, pz, qx, qy, qz, qw):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(qx, qy, qz, qw)
    T[:3, 3] = [px, py, pz]
    return T


def project_points(pts_base, K, T_base_cam, dist):
    T_cam_base = np.linalg.inv(T_base_cam)
    P = np.hstack([pts_base, np.ones((len(pts_base), 1))])
    cam = (T_cam_base @ P.T).T[:, :3]
    rvec = np.zeros(3); tvec = np.zeros(3)
    uv, _ = cv2.projectPoints(cam.reshape(-1, 1, 3), rvec, tvec,
                              np.array(K, dtype=float),
                              np.array(dist, dtype=float).reshape(-1, 1))
    return uv.reshape(-1, 2), cam[:, 2]


def receiver_bbox_at_frame(sidecar_csv, role, img_filename):
    if not os.path.exists(sidecar_csv):
        return None
    with open(sidecar_csv) as f:
        rows = [r for r in csv.DictReader(f) if r["role"] == role]
    if not rows:
        return None
    # find the row whose img_filename matches, else nearest by frame_idx
    exact = [r for r in rows if r["img_filename"] == img_filename]
    r = exact[0] if exact else rows[len(rows)//2]
    bb = [float(r["bbox_x0"]), float(r["bbox_y0"]),
          float(r["bbox_x1"]), float(r["bbox_y1"])]
    if bb[0] < 0:
        return None
    return bb, r["img_filename"]


def main():
    with open("calibration.yaml") as f:
        calib = yaml.safe_load(f)
    K = calib["camera_intrinsics"]["K"]
    dist = calib["camera_intrinsics"]["dist"]

    rows_out = []
    results = []
    for trial in TRIALS:
        df = pd.read_csv(trial["csv"])
        idx = detect_press_event(df)
        if idx is None:
            print(f"[skip] {trial['name']}: no force event", flush=True)
            continue
        r = df.iloc[idx]
        T_base_bota = pose_to_T(r[PX], r[PY], r[PZ], r[QX], r[QY], r[QZ], r[QW])
        f_b = np.array([r[FX], r[FY], r[FZ]], dtype=float)
        tau_b = np.array([r[TXc], r[TYc], r[TZc]], dtype=float)
        fmag = float(np.linalg.norm(f_b))
        f_hat_b = f_b / fmag
        r0_b = np.cross(f_b, tau_b) / (fmag**2)
        img_id = str(r[IMG])
        bgr = cv2.imread(os.path.join(trial["img_dir"], f"{img_id}.png"))
        if bgr is None:
            print(f"[skip] {trial['name']}: frame {img_id} missing", flush=True)
            continue

        recv = receiver_bbox_at_frame(trial["sidecar"], trial["receiver_role"], f"{img_id}.png")
        if recv is None:
            print(f"[skip] {trial['name']}: no receiver mask available", flush=True)
            continue
        bb, matched_img = recv
        cx, cy = (bb[0]+bb[2])/2, (bb[1]+bb[3])/2

        panel_row = [bgr.copy() for _ in CANDIDATES]
        for col, (cand_name, cand_t) in enumerate(CANDIDATES.items()):
            T_bota_cam = np.eye(4)
            T_bota_cam[:3, :3] = R_BOTA_CAM
            T_bota_cam[:3, 3] = cand_t
            T_base_cam = T_base_bota @ T_bota_cam

            r0_base = T_base_bota[:3, :3] @ r0_b + T_base_bota[:3, 3]
            f_hat_base = T_base_bota[:3, :3] @ f_hat_b
            tip = r0_base + (-f_hat_base) * 0.12
            uv, _ = project_points(np.vstack([r0_base, tip]), K, T_base_cam, dist)
            p0 = (int(uv[0][0]), int(uv[0][1]))
            p1 = (int(uv[1][0]), int(uv[1][1]))

            dist_px = float(np.hypot(p0[0]-cx, p0[1]-cy))
            results.append((trial["name"], cand_name, dist_px, p0, (cx, cy)))
            print(f"[{trial['name']}] candidate {cand_name}: ray anchor "
                  f"px={p0}  receiver_bbox_centroid={cx:.0f},{cy:.0f}  "
                  f"dist={dist_px:.0f}px", flush=True)

            ov = panel_row[col]
            cv2.rectangle(ov, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                          (0, 255, 0), 2)
            cv2.arrowedLine(ov, p0, p1, (0, 0, 255), 3, tipLength=0.25)
            cv2.circle(ov, p0, 7, (0, 0, 255), -1)
            cv2.putText(ov, f"{trial['name']} cand {cand_name}  dist={dist_px:.0f}px",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        rows_out.append(np.hstack(panel_row))

    if rows_out:
        max_w = max(r.shape[1] for r in rows_out)
        padded = [np.pad(r, ((0,0),(0, max_w-r.shape[1]),(0,0))) if r.shape[1] < max_w
                  else r for r in rows_out]
        out = np.vstack(padded)
        os.makedirs("figures", exist_ok=True)
        cv2.imwrite(OUT, out)
        print(f"\n[write] {OUT}", flush=True)

    print("\n[summary] trial, candidate, dist_px:", flush=True)
    for name, cand, d, p0, c in results:
        print(f"  {name:22s}  {cand}  {d:7.0f}px", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
