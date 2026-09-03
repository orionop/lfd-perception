"""
The shared set of contact events that carry BOTH a real wrench and a
ground-truth object mask.

Code/extrinsic_grid_search.py (which fits T_bota_camera) and
Code/wrench_ray_validate.py (which scores a given T_bota_camera) must be
evaluated on exactly the same events against exactly the same ground truth,
or a fitted transform can be made to look good simply by scoring it on a
kinder set. This module owns that list so neither can drift, the same reason
Code/dado_eval_tasks.py exists for the two label-free baselines.

Only three recordings qualify. A usable event needs a wrench topic (to define
the force line at all) and a propagated mask for the object that received the
contact. That is:

    lfdws_t001            contact_receiver, gripper + F/T
    lfdws_t001_depth      5 hand-picked activity events, 3 distinct roles
    lfdws_t001_labexport  contact_receiver, gripper present but never actuates

Seven events across three recordings is thin, and every consumer must treat it
that way: it is enough for leave-one-recording-out cross-validation and not
enough to call anything verified.

Event selection uses the guarded detector in Code/event_utils.py. That matters
here more than anywhere else: lfdws_t001_labexport's gripper spans 6.6e-7 m of
pure noise, and the unguarded midpoint rule invents a grasp window that
displaces the contact event from the true 11.15 N peak at 5.08 s to 3.34 s.
Fitting an extrinsic against a phantom event would be worse than not fitting
one at all.

Read only. Touches no pipeline output.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from event_utils import (gripper_closed_window, mask_from_overlay,
                         parse_gripper_width)

POSE_PX = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.x"
POSE_PY = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.y"
POSE_PZ = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.z"
POSE_QX = "NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation.x"
POSE_QY = "NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation.y"
POSE_QZ = "NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation.z"
POSE_QW = "NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation.w"
FX = "bota_post.wrench_body_compensated.wrench.force.x"
FY = "bota_post.wrench_body_compensated.wrench.force.y"
FZ = "bota_post.wrench_body_compensated.wrench.force.z"
TX = "bota_post.wrench_body_compensated.wrench.torque.x"
TY = "bota_post.wrench_body_compensated.wrench.torque.y"
TZ = "bota_post.wrench_body_compensated.wrench.torque.z"
GRIP = "NS_1.franka_gripper.joint_states.position"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"
# The depth stream runs on its own timeline; the merged CSV row gives the
# depth frame that matches this pose row, which is NOT the rgb frame id
# except in the lab's native export.
DEPTH_COL = "zed.zed_node.depth.depth_registered.compressedDepth"

RGB_DIR_NAME = "zed_zed_node_rgb_color_rect_image_compressed"

# Minimum force above baseline for the line of action to mean anything. Below
# this the torque is dominated by sensor noise and r0 = (f x tau)/|f|^2 is
# numerically meaningless, so the event is dropped rather than fitted against.
MIN_CONTACT_FORCE_N = 3.0

COLOR = {
    "contact_receiver": (255, 0, 255),
    "tool_contact": (0, 165, 255),
    "charger_contact": (0, 215, 255),
}

# Recordings, and how their contact events are chosen.
#   "detect" -> use the guarded detector's press event
#   explicit -> the hand-picked activity-cluster events already used by
#               Code/dado_eval_tasks.py, so the two evaluation sets agree
RECORDINGS = [
    {
        "trial": "lfdws_t001",
        "csv": "Data/lfdws_t001/lfdws_t001/lfdws_t001_0.csv",
        "rgb_dir": f"Data/lfdws_t001/lfdws_t001/{RGB_DIR_NAME}",
        "sidecar": "figures/identify/objects_summary.csv",
        "events": "detect",
        "role": "contact_receiver",
    },
    {
        "trial": "lfdws_t001_labexport",
        "csv": "Data/lfdws_t001_labexport/lfdws_t001/lfdws_t001.csv",
        "rgb_dir": f"Data/lfdws_t001_labexport/lfdws_t001/{RGB_DIR_NAME}",
        "sidecar": "figures/t001labexport/identify/objects_summary.csv",
        "events": "detect",
        "role": "contact_receiver",
    },
    {
        "trial": "lfdws_t001_depth",
        "csv": "Data/lfdws_t001_depth/lfdws_t001_depth_0.csv",
        "rgb_dir": f"Data/lfdws_t001_depth/{RGB_DIR_NAME}",
        "sidecar": "figures/identify_depth_multi/objects_summary.csv",
        "events": [
            ("plate_press",         "1782835513207923681", "contact_receiver"),
            ("screwdriver_contact", "1782835527086969733", "tool_contact"),
            ("charger_grasp",       "1782835537622884551", "charger_contact"),
            ("charger_lift",        "1782835540835169730", "charger_contact"),
            ("charger_dock",        "1782835545525039497", "charger_contact"),
        ],
    },
]


def _load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _fvec(rows, idx, keys):
    return np.array([float(rows[idx][k]) for k in keys])


def detect_press_img_id(rows):
    """Image id at the guarded press event, or None.

    Mirrors auto_seed.py / dado_eval_tasks.py: the force peak is restricted to
    the gripper-closed window when there is a real grasp cycle, and searched
    over the whole trace when the span guard says the gripper never moved.
    """
    cols = rows[0]
    if FX not in cols:
        return None
    fm = np.array([np.sqrt(float(r[FX]) ** 2 + float(r[FY]) ** 2
                           + float(r[FZ]) ** 2) for r in rows])
    baseline = float(np.median(fm[: max(1, len(fm) // 10)]))
    if GRIP in cols:
        w = np.array([parse_gripper_width(r[GRIP]) for r in rows])
        closed = gripper_closed_window(w)
        fm_adj = (np.where(closed, fm - baseline, -np.inf)
                  if closed.any() else fm - baseline)
    else:
        fm_adj = fm - baseline
    i = int(np.argmax(fm_adj))
    return str(rows[i][IMG])


def _peak_row_for_image(rows, img_id):
    """Row index within the group sharing this image id that carries the
    largest force magnitude.

    The merged CSV runs on the pose timeline, far faster than the camera, so
    one image id spans many rows. Taking the peak of the group picks the
    instant of hardest contact rather than an arbitrary sample of the group.
    """
    idxs = [i for i, r in enumerate(rows) if str(r[IMG]) == img_id]
    if not idxs:
        return None
    fm = [np.sqrt(float(rows[i][FX]) ** 2 + float(rows[i][FY]) ** 2
                  + float(rows[i][FZ]) ** 2) for i in idxs]
    return idxs[int(np.argmax(fm))]


def _gt_mask(sidecar_rows, role, img_filename, rgb_path):
    """Ground-truth mask, with a documented fallback.

    mask_from_overlay is lossy on bright objects (a 0.5 alpha blend saturates
    where the source is already near 255), so when recovery comes back empty
    the bbox rectangle from the sidecar is used instead. That is coarser but
    honest, and it is flagged in the returned record so consumers can report
    how many events fell back.
    """
    hit = [r for r in sidecar_rows
           if r["role"] == role and r["img_filename"] == img_filename]
    if not hit:
        return None, None
    row = hit[0]
    m = mask_from_overlay(row["overlay_path"], rgb_path, COLOR[role])
    if m is not None and m.sum() > 0:
        return m, "overlay"
    try:
        x0, y0 = int(float(row["bbox_x0"])), int(float(row["bbox_y0"]))
        x1, y1 = int(float(row["bbox_x1"])), int(float(row["bbox_y1"]))
    except (ValueError, KeyError):
        return None, None
    if x0 < 0 or x1 <= x0 or y1 <= y0:
        return None, None
    import cv2
    src = cv2.imread(rgb_path)
    if src is None:
        return None, None
    m = np.zeros(src.shape[:2], dtype=bool)
    m[y0:y1 + 1, x0:x1 + 1] = True
    return m, "bbox"


def build_events(verbose=True):
    """List of fully resolved contact events.

    Each entry carries everything a consumer needs, so neither the search nor
    the validator re-derives geometry and they cannot disagree:
        trial, event, role, img_id, rgb_path,
        pose (7 floats), force (3), torque (3), force_mag,
        mask (bool HxW), mask_source ('overlay' or 'bbox')
    """
    import cv2
    out = []
    for rec in RECORDINGS:
        trial = rec["trial"]
        if not os.path.exists(rec["csv"]):
            if verbose:
                print(f"[skip] {trial}: merged CSV not found", flush=True)
            continue
        if not os.path.exists(rec["sidecar"]):
            if verbose:
                print(f"[skip] {trial}: sidecar not found", flush=True)
            continue
        rows = _load_rows(rec["csv"])
        side = _load_rows(rec["sidecar"])
        if FX not in rows[0]:
            if verbose:
                print(f"[skip] {trial}: no wrench topic", flush=True)
            continue

        if rec["events"] == "detect":
            img_id = detect_press_img_id(rows)
            spec = [] if img_id is None else [("press", img_id, rec["role"])]
        else:
            spec = rec["events"]

        for event, img_id, role in spec:
            idx = _peak_row_for_image(rows, img_id)
            if idx is None:
                if verbose:
                    print(f"[skip] {trial}/{event}: image id not in CSV",
                          flush=True)
                continue
            rgb_path = os.path.join(rec["rgb_dir"], f"{img_id}.png")
            if not os.path.exists(rgb_path):
                if verbose:
                    print(f"[skip] {trial}/{event}: rgb missing", flush=True)
                continue
            f = _fvec(rows, idx, (FX, FY, FZ))
            tau = _fvec(rows, idx, (TX, TY, TZ))
            fmag = float(np.linalg.norm(f))
            if fmag < MIN_CONTACT_FORCE_N:
                if verbose:
                    print(f"[skip] {trial}/{event}: |F|={fmag:.2f} N below "
                          f"{MIN_CONTACT_FORCE_N} N, line of action is noise",
                          flush=True)
                continue
            mask, src_kind = _gt_mask(side, role, f"{img_id}.png", rgb_path)
            if mask is None or not mask.any():
                if verbose:
                    print(f"[skip] {trial}/{event}: no ground-truth mask",
                          flush=True)
                continue
            img = cv2.imread(rgb_path)
            H, W = img.shape[:2]
            depth_id = (str(rows[idx][DEPTH_COL])
                        if DEPTH_COL in rows[idx] else img_id)
            out.append({
                "trial": trial, "event": event, "role": role,
                "img_id": img_id, "depth_id": depth_id, "rgb_path": rgb_path,
                "pose": _fvec(rows, idx, (POSE_PX, POSE_PY, POSE_PZ,
                                          POSE_QX, POSE_QY, POSE_QZ, POSE_QW)),
                "force": f, "torque": tau, "force_mag": fmag,
                "mask": mask, "mask_source": src_kind,
                "H": H, "W": W,
                "mask_frac": float(mask.sum()) / float(H * W),
            })
            if verbose:
                print(f"[event] {trial}/{event:20s} role={role:16s} "
                      f"|F|={fmag:6.2f} N  mask={int(mask.sum()):7d} px "
                      f"({out[-1]['mask_frac']:.3%} of frame, "
                      f"via {src_kind})", flush=True)
    return out


if __name__ == "__main__":
    evs = build_events()
    print(f"\n[summary] {len(evs)} contact events across "
          f"{len(set(e['trial'] for e in evs))} recordings", flush=True)
    for t in sorted(set(e["trial"] for e in evs)):
        n = sum(1 for e in evs if e["trial"] == t)
        print(f"  {t:24s} {n} event(s)", flush=True)
