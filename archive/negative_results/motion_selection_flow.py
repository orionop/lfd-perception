"""
Image stationarity under known ego-motion, as an actual SELECTION rule.

THE CLAIM BEING TESTED
----------------------
Code/baseline_sam_depth_ranking.py established that label-free discovery is
not limited by segmentation: SAM's proposal set contains the right object
(oracle IoU 0.612 mean, 0.787 median, 0.973 max) while depth ranking recovers
only 0.104. 83 percent of achievable performance is lost at SELECTION, and 41
percent of proposals sit within 2 cm of the target in depth, so depth provably
cannot break the tie. DINOv2 attention times depth does no better (0.166).

The proposal here is that proprioception plus the eye-in-hand geometry
supplies the missing selection signal, with NO camera extrinsic required:

  the camera is bolted to the same body as the gripper, so an object that is
  genuinely held is rigidly attached to the camera and must stay put in the
  image however far the arm travels, while every static world object sweeps
  across the frame under camera ego-motion.

Code/motion_selection_probe.py found the effect on 3 of 5 trials using bbox
centroids as a proxy (12x to 28x drop in image motion per metre travelled,
94-98 percent of held frames under 2 px). This script replaces the proxy with
real dense optical flow and, more importantly, turns it into a rule that picks
one proposal out of SAM's set, scored on exactly the same events as the two
existing baselines via Code/dado_eval_tasks.py.

WHERE IT APPLIES, AND WHERE IT HONESTLY DOES NOT
------------------------------------------------
Stationarity identifies the HELD object. It says nothing about a contact
receiver, which is world-static and therefore flows with the background like
everything else. So results are reported split:

    grasped-role events      the mechanism applies, this is the real number
    contact-role events      reported for completeness, expected to fail

Quoting the pooled number alone would overstate the result, so the split is
printed first and the pooled figure second.

METHOD
------
For each event frame, find a later frame separated by at least
FLOW_MIN_TRAVEL_M of end effector travel, taken from current_pose. That
matters: flow between two frames where the arm did not move carries no
information, and picking the second frame by a fixed index would silently do
that on slow segments. Compute dense Farneback flow between the pair, run the
same SAM automatic mask generator and the same two filters the depth baseline
uses (MIN_AREA, drop border-touching proposals), then select the proposal with
the LOWEST median flow magnitude. Score IoU against the propagated ground
truth, and record the oracle for comparability.

Runs SAM on CPU: SamAutomaticMaskGenerator builds float64 point grids that MPS
rejects, the same reason Code/auto_seed.py and the depth baseline force CPU.

Usage:
    .venv_sam2/bin/python Code/motion_selection_flow.py
"""
import csv
import os
import shutil
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from event_utils import mask_from_overlay

CKPT = "sam_vit_h_4b8939.pth"
MODEL = "vit_h"
OUT_CSV = "figures/motion_selection_flow.csv"
OUT_PNG = "figures/motion_selection_flow.png"

POSE_X = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.x"
POSE_Y = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.y"
POSE_Z = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.z"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"

# Same two filters as Code/baseline_sam_depth_ranking.py, so the comparison is
# like for like rather than a differently-tuned strawman.
MIN_AREA = 500

# Minimum end effector travel between the flow pair. Below this the camera has
# barely moved and background flow is indistinguishable from a held object.
FLOW_MIN_TRAVEL_M = 0.015
# Do not look further ahead than this many frames for that travel.
FLOW_MAX_GAP = 90

TRIAL_CSV = {
    "lfdws_t001_depth": "Data/lfdws_t001_depth/lfdws_t001_depth_0.csv",
    "lfdws_t002_new": "Data/lfdws_t002_new/lfdws_t002_new_0.csv",
    "lfdws_t001_labexport": "Data/lfdws_t001_labexport/lfdws_t001/lfdws_t001.csv",
    "lfdws_t004": "Data/lfdws_t004/lfdws_t004_0.csv",
    "lfdws_t005": "Data/lfdws_t005/lfdws_t005_0.csv",
}

GRASPED_ROLES = {"grasped"}


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum()) / float(u) if u else float("nan")


def frame_pose_index(csv_path):
    """{image_id: ee_xyz} using the first pose row that carries each image.

    The merged CSV runs on the pose timeline, far faster than the camera, so
    one image id spans many rows; the first is enough to locate the camera.
    """
    out = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            iid = str(r[IMG])
            if iid in out:
                continue
            try:
                out[iid] = np.array([float(r[POSE_X]), float(r[POSE_Y]),
                                     float(r[POSE_Z])])
            except (ValueError, KeyError):
                continue
    return out


def pick_flow_partner(frame_ids, poses, i):
    """Index of a later frame at least FLOW_MIN_TRAVEL_M away, or None."""
    p0 = poses.get(frame_ids[i])
    if p0 is None:
        return None, 0.0
    best = None
    for j in range(i + 1, min(i + FLOW_MAX_GAP + 1, len(frame_ids))):
        p1 = poses.get(frame_ids[j])
        if p1 is None:
            continue
        d = float(np.linalg.norm(p1 - p0))
        best = (j, d)
        if d >= FLOW_MIN_TRAVEL_M:
            return j, d
    return (best[0], best[1]) if best else (None, 0.0)


def main():
    dev = pick_device()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dado_eval_tasks import TASKS
    print(f"[setup] device={dev}  {len(TASKS)} events "
          f"(identical set to both existing baselines)", flush=True)
    print(f"[setup] flow pair needs >= {FLOW_MIN_TRAVEL_M*1000:.0f} mm of end "
          f"effector travel, searched up to {FLOW_MAX_GAP} frames ahead",
          flush=True)

    print(f"[load] SAM {MODEL} from {CKPT}", flush=True)
    sam = sam_model_registry[MODEL](checkpoint=CKPT).to(dev)
    gen = SamAutomaticMaskGenerator(sam, points_per_side=16,
                                    pred_iou_thresh=0.86,
                                    stability_score_thresh=0.90)

    pose_cache, dir_cache, sidecars = {}, {}, {}
    rows, panels = [], []

    for k, (trial, event, role, color, rgb_p, depth_p, side_csv, img_id) in \
            enumerate(TASKS):
        tag = f"{trial}/{event}"
        bgr = cv2.imread(rgb_p)
        if bgr is None:
            print(f"[skip] {tag}: rgb unreadable", flush=True)
            continue
        H, W = bgr.shape[:2]

        rgb_dir = os.path.dirname(rgb_p)
        if rgb_dir not in dir_cache:
            ids = sorted(os.path.splitext(f)[0]
                         for f in os.listdir(rgb_dir) if f.endswith(".png"))
            dir_cache[rgb_dir] = ids
        frame_ids = dir_cache[rgb_dir]

        cpath = TRIAL_CSV.get(trial)
        if cpath is None or not os.path.exists(cpath):
            print(f"[skip] {tag}: no merged CSV", flush=True)
            continue
        if trial not in pose_cache:
            pose_cache[trial] = frame_pose_index(cpath)
        poses = pose_cache[trial]

        try:
            i = frame_ids.index(str(img_id))
        except ValueError:
            print(f"[skip] {tag}: event frame not in rgb dir", flush=True)
            continue
        j, travel = pick_flow_partner(frame_ids, poses, i)
        if j is None:
            print(f"[skip] {tag}: no later frame with pose", flush=True)
            continue
        if travel < FLOW_MIN_TRAVEL_M:
            print(f"[warn] {tag}: only {travel*1000:.1f} mm of travel "
                  f"available within {FLOW_MAX_GAP} frames, flow will be weak",
                  flush=True)

        nxt = cv2.imread(os.path.join(rgb_dir, f"{frame_ids[j]}.png"))
        if nxt is None:
            print(f"[skip] {tag}: partner frame unreadable", flush=True)
            continue

        if side_csv not in sidecars:
            sidecars[side_csv] = list(csv.DictReader(open(side_csv))) \
                if os.path.exists(side_csv) else []
        gt_rows = [r for r in sidecars[side_csv]
                   if r["role"] == role and r["img_filename"] == f"{img_id}.png"]
        if not gt_rows:
            print(f"[skip] {tag}: no ground-truth row", flush=True)
            continue
        gt = mask_from_overlay(gt_rows[0]["overlay_path"], rgb_p, color)
        if gt is None or gt.sum() == 0:
            print(f"[skip] {tag}: empty ground truth", flush=True)
            continue

        print(f"[run] {tag}: flow over {j-i} frames / {travel*1000:.1f} mm, "
              f"then SAM proposals ...", flush=True)
        g0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 21, 3,
                                            5, 1.2, 0)
        fmag = np.linalg.norm(flow, axis=2)
        bg_med = float(np.median(fmag))

        anns = gen.generate(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not anns:
            print(f"  [warn] no proposals", flush=True)
            continue

        border = np.zeros((H, W), dtype=bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True

        cands = []
        for a in anns:
            m = a["segmentation"]
            if m.sum() < MIN_AREA or (m & border).any():
                continue
            cands.append((float(np.median(fmag[m])), m))
        if not cands:
            print(f"  [warn] all proposals filtered out", flush=True)
            continue

        cands.sort(key=lambda t: t[0])
        pick_flow, pick = cands[0]
        achieved = iou(pick, gt)
        oracle = max(iou(m, gt) for _, m in cands)
        gt_flow = float(np.median(fmag[gt]))

        print(f"  [result] {tag}: IoU={achieved:.3f}  oracle={oracle:.3f}  "
              f"picked flow={pick_flow:5.2f} px  gt flow={gt_flow:5.2f} px  "
              f"background median={bg_med:5.2f} px  "
              f"({len(cands)}/{len(anns)} proposals survived)", flush=True)

        rows.append([trial, event, role, "grasped" if role in GRASPED_ROLES
                     else "contact", f"{achieved:.4f}", f"{oracle:.4f}",
                     f"{pick_flow:.3f}", f"{gt_flow:.3f}", f"{bg_med:.3f}",
                     len(cands), len(anns), j - i, f"{travel:.4f}"])

        vis = bgr.copy()
        cont, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cont, -1, (0, 255, 0), 2)
        cont, _ = cv2.findContours(pick.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cont, -1, (0, 0, 220), 2)
        cv2.putText(vis, f"{event} IoU={achieved:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        panels.append(vis)

    if not rows:
        print("[fatal] no events scored", flush=True)
        return

    arr = {kind: [float(r[4]) for r in rows if r[3] == kind]
           for kind in ("grasped", "contact")}
    orc = {kind: [float(r[5]) for r in rows if r[3] == kind]
           for kind in ("grasped", "contact")}

    print("\n[summary] stationarity selection, split by whether the "
          "mechanism applies", flush=True)
    for kind in ("grasped", "contact"):
        v, o = arr[kind], orc[kind]
        if not v:
            print(f"  {kind:8s}: no events", flush=True)
            continue
        note = ("the mechanism applies" if kind == "grasped"
                else "world-static object, mechanism NOT expected to work")
        print(f"  {kind:8s}: n={len(v):2d}  mean IoU={np.mean(v):.3f}  "
              f"oracle={np.mean(o):.3f}   ({note})", flush=True)

    allv = [float(r[4]) for r in rows]
    print(f"\n  pooled over all {len(allv)} events: mean IoU="
          f"{np.mean(allv):.3f}", flush=True)
    print("  [compare] SAM automask + depth ranking : 0.104", flush=True)
    print("  [compare] DINOv2 attention x depth     : 0.166", flush=True)
    print("  Pooled is the weaker way to read this: it averages in the "
          "contact events where\n  stationarity cannot work by construction. "
          "The grasped row is the real number.", flush=True)

    os.makedirs("figures", exist_ok=True)
    if os.path.exists(OUT_CSV):
        shutil.copy2(OUT_CSV, OUT_CSV + ".bak")
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "event", "role", "kind", "iou", "oracle_iou",
                    "picked_flow_px", "gt_flow_px", "bg_median_flow_px",
                    "n_proposals_kept", "n_proposals_raw", "frame_gap",
                    "ee_travel_m"])
        w.writerows(rows)
    print(f"[write] {OUT_CSV}", flush=True)

    if panels:
        h = min(p.shape[0] for p in panels)
        res = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h))
               for p in panels]
        per_row = 4
        grid = []
        for i in range(0, len(res), per_row):
            ch = res[i:i + per_row]
            wmax = max(c.shape[1] for c in ch)
            ch = [cv2.copyMakeBorder(c, 0, 0, 0, wmax - c.shape[1],
                                     cv2.BORDER_CONSTANT, value=(20, 20, 20))
                  for c in ch]
            while len(ch) < per_row:
                ch.append(np.full_like(ch[0], 20))
            grid.append(np.hstack(ch))
        wmax = max(g.shape[1] for g in grid)
        grid = [cv2.copyMakeBorder(g, 0, 0, 0, wmax - g.shape[1],
                                   cv2.BORDER_CONSTANT, value=(20, 20, 20))
                for g in grid]
        if os.path.exists(OUT_PNG):
            shutil.copy2(OUT_PNG, OUT_PNG + ".bak")
        cv2.imwrite(OUT_PNG, np.vstack(grid))
        print(f"[write] {OUT_PNG}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
