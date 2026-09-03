"""
Force-direction sanity check (uncalibrated).

At the press event:
  - take the wrench in robot base frame (from the merged CSV)
  - compute force direction in base coords
  - estimate the image-plane direction of the force by fitting a simple
    base->image mapping from the data itself:
        * we know the END-EFFECTOR position in base coords over time (pose)
        * we have the carrot-mask centroid over time (from the propagation CSV)
        * regress (base_x, base_y, base_z) -> (u, v) via least squares
        (this is an UNCALIBRATED approximation, NOT real intrinsics+extrinsics)
  - draw an arrow from the gripper position in the press frame, in the
    direction of the negative force vector, projected through the fitted map
  - overlay the cup mask (magenta) and the carrot mask (green)

The point: SEE whether the wrench direction at press actually points at the
cup. If yes, the wrench-projection idea is alive on this data even without
real camera calibration. If not, we honestly say so.

Requires a GRASPED object (carrot) trajectory to fit the base->uv
regression -- on a trial with no gripper topic (no grasp event, no
propagate_demo.py run) this has nothing to fit against and exits early
with "[fatal] too few pairs to fit projection". That's a real data gap
(see Docs/FAILURE_MODES.md B3), not a bug.

Output:
    figures/force_overlay_press.png  (or --out)

Usage:
    .venv_analysis/bin/python Code/force_overlay.py \
        --trial Data/lfdws_t001/lfdws_t001 \
        --carrot_csv figures/propagation_summary.csv \
        --cup_csv figures/propagation_cup_summary.csv
"""
import argparse
import ast
import csv
import os

import cv2
import numpy as np

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from event_utils import gripper_moved
import pandas as pd

IMG_DIR_NAME = "zed_zed_node_rgb_color_rect_image_compressed"

POSE_TS = "NS_1.franka_robot_state_broadcaster.current_pose.timestamp"
GRIP = "NS_1.franka_gripper.joint_states.position"
PX, PY, PZ = (
    "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.x",
    "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.y",
    "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.z",
)
FX = "bota_post.wrench_body_compensated.wrench.force.x"
FY = "bota_post.wrench_body_compensated.wrench.force.y"
FZ = "bota_post.wrench_body_compensated.wrench.force.z"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"


def mask_centroid_from_overlay(overlay_path, src_path, color):
    """Crude: find pixels in overlay much greener (or magenta) than source."""
    ov = cv2.imread(overlay_path)
    src = cv2.imread(src_path)
    if ov is None or src is None or ov.shape != src.shape:
        return None
    diff = ov.astype(int) - src.astype(int)
    if color == "green":
        mask = diff[..., 1] > 40
    elif color == "magenta":
        mask = (diff[..., 0] > 40) & (diff[..., 2] > 40)
    else:
        return None
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def detect_press_row(df):
    """Same force-only-safe press detection as auto_seed.py: strongest
    force-magnitude peak, restricted to the gripper-held window if a
    gripper topic is present, else over the whole trace."""
    fm = np.sqrt(df[FX].astype(float) ** 2 + df[FY].astype(float) ** 2 +
                 df[FZ].astype(float) ** 2).to_numpy()
    baseline = float(np.median(fm[: len(fm) // 10]))
    if GRIP not in df.columns:
        return int(np.argmax(fm - baseline))

    def parse_gw(c):
        try:
            return float(np.sum(ast.literal_eval(c)))
        except Exception:
            return float("nan")
    w = df[GRIP].apply(parse_gw).to_numpy()
    w_open, w_closed = float(np.nanmax(w)), float(np.nanmin(w))
    thr = w_closed + 0.5 * (w_open - w_closed)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", default="Data/lfdws_t001/lfdws_t001")
    ap.add_argument("--carrot_csv", default="figures/propagation_summary.csv")
    ap.add_argument("--cup_csv", default="figures/propagation_cup_summary.csv")
    ap.add_argument("--out", default="figures/force_overlay_press.png")
    args = ap.parse_args()

    demo_csv = next((os.path.join(args.trial, f) for f in os.listdir(args.trial)
                     if f.endswith(".csv") and not f.startswith(".")), None)
    if demo_csv is None:
        raise FileNotFoundError(f"no merged CSV in {args.trial}")
    img_dir = os.path.join(args.trial, IMG_DIR_NAME)

    print(f"[load] {demo_csv}", flush=True)
    df = pd.read_csv(demo_csv)

    if FX not in df.columns:
        print("[fatal] no wrench topic in this trial -- nothing to compute "
              "a force direction from (see Docs/FAILURE_MODES.md)", flush=True)
        return

    # ---------- press-frame index ----------
    press_row = detect_press_row(df)
    press_img_ts = str(df[IMG].iloc[press_row])
    print(f"[load] press row idx = {press_row}  img={press_img_ts}", flush=True)

    # ---------- EE positions across demo ----------
    ee = df[[PX, PY, PZ]].astype(float).to_numpy()  # (N, 3)
    img_ids = df[IMG].astype(str).to_numpy()
    print(f"[load] EE trajectory: {ee.shape}", flush=True)

    # ---------- build (EE_base) -> (u, v) regression from carrot mask centroids ----------
    # for each frame where we have a carrot mask, find the EE position
    print(f"[load] carrot summary: {args.carrot_csv}", flush=True)
    pairs = []
    if os.path.exists(args.carrot_csv):
        with open(args.carrot_csv) as f:
            for row in csv.DictReader(f):
                img_id = row["file"].replace(".png", "")
                ov_path = row["overlay_path"]
                src_path = os.path.join(img_dir, row["file"])
                c = mask_centroid_from_overlay(ov_path, src_path, "green")
                if c is None:
                    continue
                idxs = np.where(img_ids == img_id)[0]
                if len(idxs) == 0:
                    continue
                ee_pos = ee[int(idxs[0])]
                pairs.append((ee_pos, c))
    print(f"[fit] collected {len(pairs)} (EE, mask-centroid) pairs", flush=True)
    if len(pairs) < 6:
        print("[fatal] too few pairs to fit projection", flush=True)
        return

    A = np.array([np.append(p[0], 1) for p in pairs])  # (N, 4)  [x, y, z, 1]
    UV = np.array([p[1] for p in pairs])                # (N, 2)
    # least-squares: A @ M = UV  -> M = (4, 2)
    M, *_ = np.linalg.lstsq(A, UV, rcond=None)
    pred = A @ M
    rmse = float(np.sqrt(((pred - UV) ** 2).sum(axis=1).mean()))
    print(f"[fit] linear base->uv RMSE = {rmse:.1f}px (over {len(pairs)} pts)", flush=True)

    def project(x, y, z):
        return tuple((np.array([x, y, z, 1.0]) @ M).tolist())

    # ---------- press: EE pos + force in base ----------
    ee_press = ee[press_row]
    # wrench: bota frame ≈ wrist; we project the *displacement* base+(-F) -> base+(-F+step)
    f = np.array([df[FX].iloc[press_row], df[FY].iloc[press_row], df[FZ].iloc[press_row]],
                 dtype=float)
    f_mag = float(np.linalg.norm(f))
    f_hat = f / f_mag if f_mag > 1e-3 else np.zeros(3)
    print(f"[press] EE = {ee_press}   F = {f}  |F| = {f_mag:.2f} N", flush=True)

    # tail and head of the arrow in base coords
    tail_base = ee_press
    head_base = ee_press + (-f_hat) * 0.10  # 10 cm along negative force (into contact)
    tail_uv = project(*tail_base)
    head_uv = project(*head_base)
    print(f"[press] arrow uv: {tail_uv} -> {head_uv}", flush=True)

    # ---------- render ----------
    src_path = os.path.join(img_dir, f"{press_img_ts}.png")
    base = cv2.imread(src_path)
    if base is None:
        print(f"[fatal] {src_path} missing", flush=True); return
    H, W = base.shape[:2]
    out = base.copy()

    # overlay carrot mask (green) and cup mask (magenta)
    for path, color in [(args.carrot_csv, "green"), (args.cup_csv, "magenta")]:
        if not os.path.exists(path):
            continue
        with open(path) as f_:
            for row in csv.DictReader(f_):
                if row["file"] != f"{press_img_ts}.png":
                    continue
                ov = cv2.imread(row["overlay_path"])
                if ov is None or ov.shape != base.shape:
                    continue
                diff = ov.astype(int) - base.astype(int)
                if color == "green":
                    m = diff[..., 1] > 40
                    rgb = (0, 255, 0)
                else:
                    m = (diff[..., 0] > 40) & (diff[..., 2] > 40)
                    rgb = (255, 0, 255)
                layer = np.zeros_like(out)
                layer[m] = rgb
                out = cv2.addWeighted(out, 1.0, layer, 0.4, 0)

    def clip(uv):
        return (int(np.clip(uv[0], 0, W - 1)), int(np.clip(uv[1], 0, H - 1)))

    cv2.arrowedLine(out, clip(tail_uv), clip(head_uv), (0, 0, 255), 4, tipLength=0.25)
    cv2.circle(out, clip(tail_uv), 6, (255, 255, 255), -1)
    cv2.putText(out, f"-F dir  (|F|={f_mag:.1f} N)  proj-RMSE={rmse:.0f}px",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(out, "carrot (green)  cup (magenta)  EE base->uv via least-squares fit",
                (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, out)
    print(f"[save] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
