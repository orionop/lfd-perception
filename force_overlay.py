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

Output:
    figures/force_overlay_press.png
"""
import ast
import csv
import os

import cv2
import numpy as np
import pandas as pd

DEMO_CSV = "lfdws_t001/lfdws_t001/lfdws_t001_0.csv"
IMG_DIR = "lfdws_t001/lfdws_t001/zed_zed_node_rgb_color_rect_image_compressed"
PRESS_IMG_TS = 1779192196405413163
CARROT_SUMMARY = "figures/propagation_summary.csv"
CUP_SUMMARY = "figures/propagation_cup_summary.csv"
OUT = "figures/force_overlay_press.png"

POSE_TS = "NS_1.franka_robot_state_broadcaster.current_pose.timestamp"
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


def main():
    print(f"[load] {DEMO_CSV}", flush=True)
    df = pd.read_csv(DEMO_CSV)

    # ---------- press-frame index ----------
    press_str = str(PRESS_IMG_TS)
    press_rows = df.index[df[IMG].astype(str) == press_str]
    if len(press_rows) == 0:
        print(f"[fatal] press img ts {press_str} not in CSV", flush=True)
        return
    press_row = int(press_rows[0])
    print(f"[load] press row idx = {press_row}", flush=True)

    # ---------- EE positions across demo ----------
    ee = df[[PX, PY, PZ]].astype(float).to_numpy()  # (N, 3)
    img_ids = df[IMG].astype(str).to_numpy()
    print(f"[load] EE trajectory: {ee.shape}", flush=True)

    # ---------- build (EE_base) -> (u, v) regression from carrot mask centroids ----------
    # for each frame where we have a carrot mask, find the EE position
    print(f"[load] carrot summary: {CARROT_SUMMARY}", flush=True)
    pairs = []
    if os.path.exists(CARROT_SUMMARY):
        with open(CARROT_SUMMARY) as f:
            for row in csv.DictReader(f):
                img_id = row["file"].replace(".png", "")
                ov_path = row["overlay_path"]
                src_path = os.path.join(IMG_DIR, row["file"])
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
    src_path = os.path.join(IMG_DIR, f"{PRESS_IMG_TS}.png")
    base = cv2.imread(src_path)
    if base is None:
        print(f"[fatal] {src_path} missing", flush=True); return
    H, W = base.shape[:2]
    out = base.copy()

    # overlay carrot mask (green) and cup mask (magenta)
    for path, color in [(CARROT_SUMMARY, "green"), (CUP_SUMMARY, "magenta")]:
        if not os.path.exists(path):
            continue
        with open(path) as f_:
            for row in csv.DictReader(f_):
                if row["file"] != f"{PRESS_IMG_TS}.png":
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

    os.makedirs("figures", exist_ok=True)
    cv2.imwrite(OUT, out)
    print(f"[save] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
