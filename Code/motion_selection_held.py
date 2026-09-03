"""
Image stationarity as a selection rule, retested properly.

WHY A RETEST
------------
Code/motion_selection_flow.py scored 0.037 mean IoU, worse than both existing
baselines (SAM automask + depth 0.104, DINOv2 attention x depth 0.166). The
per-event diagnostics showed the failure was in the experiment, not obviously
in the hypothesis, and named two causes:

  1. WRONG FRAMES. It scored on Code/dado_eval_tasks.py's events for
     comparability, but those are the grasp, release and press INSTANTS. At
     those moments the object is in transition, not rigidly held. lfdws_t002_new
     /grasp showed the object flowing at 9.95 px against a 6.55 px background,
     moving MORE than the scene, because the gripper is still closing. The
     mechanism only applies across the held window, which is what
     Code/motion_selection_probe.py measured when it found ratios of
     0.023-0.051 and 95-100 percent of held frames static.

  2. DEGENERATE DECISION RULE. Picking argmin over per-proposal flow selects
     whatever region is most static, and Farneback returns near-zero flow on
     any textureless patch. On lfdws_t005 the signal was plainly present (the
     true object at 1.13 px against a 4.38 px background, a 3.9x separation)
     and the rule still picked a blank region at 0.39 px.

A third defect surfaced in Code/motion_selection_diagnose.py: lfdws_t004's
gripper-closed mask is not one grasp. Its per-segment median width swings
18 -> 34 -> 18 -> 6 mm, so several cycles on differently-sized objects are
merged into a single "held" window, and the tracked object is only one of them.

WHAT THIS SCRIPT CHANGES
------------------------
  * Splits the closed mask into CONTIGUOUS runs and treats each as its own
    grasp cycle, so lfdws_t004 stops being an invalid test.
  * Samples frames from INSIDE each hold rather than at its boundaries.
  * Gates proposals on texture before ranking, using an adaptive threshold
    (the median texture of the surviving proposals) so no magic constant is
    tuned to a trial.
  * Requires that the camera actually moved between the flow pair, and skips
    the event outright when it did not, instead of scoring an undecidable
    frame.

HONESTY ABOUT COMPARABILITY
---------------------------
This is a DIFFERENT evaluation set from the 11-event one, so the 0.104 and
0.166 figures cannot be quoted against it directly. To keep the comparison
fair, every baseline is recomputed on THIS set from the same proposal pool:

    random     mean IoU over all surviving proposals, the exact expectation
               of a uniformly random pick. This is the null.
    largest    pick the biggest surviving proposal
    central    pick the proposal closest to the image centre
    depth      nearest-median-depth, where the trial has real depth
    oracle     best IoU any surviving proposal achieves, the ceiling

A result only counts if it beats `random` clearly. Beating nothing else would
mean the proposal pool, not the rule, is doing the work.

Usage:
    .venv_sam2/bin/python Code/motion_selection_held.py
    .venv_sam2/bin/python Code/motion_selection_held.py --per_cycle 2
"""
import argparse
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
from event_utils import (gripper_closed_window, mask_from_overlay,
                         parse_gripper_width)

CKPT = "sam_vit_h_4b8939.pth"
MODEL = "vit_h"
OUT_CSV = "figures/motion_selection_held.csv"
OUT_PNG = "figures/motion_selection_held.png"

POSE_X = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.x"
POSE_Y = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.y"
POSE_Z = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.z"
GRIP = "NS_1.franka_gripper.joint_states.position"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"
DEPTH_COL = "zed.zed_node.depth.depth_registered.compressedDepth"

MIN_AREA = 500
FLOW_MIN_TRAVEL_M = 0.015
FLOW_MAX_GAP = 90
# Below this the camera has barely moved and nothing is separable from a held
# object, so the event is skipped rather than scored.
MIN_BG_FLOW_PX = 1.0
# A contiguous closed run shorter than this is not a real grasp cycle.
MIN_CYCLE_FRAMES = 30

GRASPED_COLOR = (0, 255, 0)

TRIALS = [
    # label, merged csv, rgb dir, depth dir or None, sidecar, role
    ("lfdws_t001",
     "Data/lfdws_t001/lfdws_t001/lfdws_t001_0.csv",
     "Data/lfdws_t001/lfdws_t001/zed_zed_node_rgb_color_rect_image_compressed",
     None,
     "figures/identify/objects_summary.csv", "grasped"),
    ("lfdws_t002_new",
     "Data/lfdws_t002_new/lfdws_t002_new_0.csv",
     "Data/lfdws_t002_new/zed_zed_node_rgb_color_rect_image_compressed",
     "Data/lfdws_t002_new/zed_zed_node_depth_depth_registered_compressedDepth",
     "figures/t002new/identify/objects_summary.csv", "grasped"),
    ("lfdws_t002_labexport",
     "Data/lfdws_t002_labexport/lfdws_t002/lfdws_t002.csv",
     "Data/lfdws_t002_labexport/lfdws_t002/zed_zed_node_rgb_color_rect_image_compressed",
     "Data/lfdws_t002_labexport/lfdws_t002/zed_zed_node_depth_depth_registered_compressedDepth",
     "figures/t002labexport/identify/objects_summary.csv", "grasped"),
    ("lfdws_t004",
     "Data/lfdws_t004/lfdws_t004_0.csv",
     "Data/lfdws_t004/zed_zed_node_rgb_color_rect_image_compressed",
     "Data/lfdws_t004/zed_zed_node_depth_depth_registered_compressedDepth",
     "figures/t004/identify/objects_summary.csv", "grasped"),
    ("lfdws_t005",
     "Data/lfdws_t005/lfdws_t005_0.csv",
     "Data/lfdws_t005/zed_zed_node_rgb_color_rect_image_compressed",
     "Data/lfdws_t005/zed_zed_node_depth_depth_registered_compressedDepth",
     "figures/t005/identify/objects_summary.csv", "grasped"),
]


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum()) / float(u) if u else float("nan")


def contiguous_runs(mask, min_len):
    """[(start, end)) for each contiguous True run at least min_len long."""
    runs, i, n = [], 0, len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if j - i >= min_len:
            runs.append((i, j))
        i = j
    return runs


def load_trial(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return rows


def build_held_events(args):
    """Frames sampled from INSIDE each grasp cycle, one list per trial."""
    events = []
    for (label, cpath, rgb_dir, depth_dir, sidecar, role) in TRIALS:
        if not (os.path.exists(cpath) and os.path.exists(sidecar)):
            print(f"[skip] {label}: missing csv or sidecar", flush=True)
            continue
        rows = load_trial(cpath)
        if GRIP not in rows[0]:
            print(f"[skip] {label}: no gripper topic", flush=True)
            continue
        w = np.array([parse_gripper_width(r[GRIP]) for r in rows])
        closed = gripper_closed_window(w)
        if not closed.any():
            print(f"[skip] {label}: span guard, gripper never actuated",
                  flush=True)
            continue

        # collapse the pose timeline to one entry per distinct image frame
        frames, seen = [], set()
        for i, r in enumerate(rows):
            iid = str(r[IMG])
            if iid in seen:
                continue
            seen.add(iid)
            frames.append({
                "img_id": iid,
                "ee": np.array([float(r[POSE_X]), float(r[POSE_Y]),
                                float(r[POSE_Z])]),
                "closed": bool(closed[i]),
                "width": float(w[i]),
                "depth_id": str(r[DEPTH_COL]) if DEPTH_COL in r else iid,
            })
        fclosed = np.array([f["closed"] for f in frames])
        runs = contiguous_runs(fclosed, MIN_CYCLE_FRAMES)
        widths = [float(np.nanmedian([frames[k]["width"]
                                      for k in range(a, b)])) for a, b in runs]
        print(f"[cycles] {label}: {len(runs)} grasp cycle(s) of >= "
              f"{MIN_CYCLE_FRAMES} frames; median widths (mm) "
              f"{[round(x*1000,1) for x in widths]}", flush=True)

        side = list(csv.DictReader(open(sidecar)))
        track = {os.path.splitext(r["img_filename"])[0]: r
                 for r in side if r["role"] == role
                 and float(r["mask_px"]) > 0}

        for ci, (a, b) in enumerate(runs):
            # sample from inside the hold, avoiding the first and last 15%
            span = b - a
            lo, hi = a + int(0.15 * span), b - int(0.15 * span)
            if hi - lo < 4:
                continue
            picks = np.linspace(lo, hi - 1, args.per_cycle + 2)[1:-1]
            for p in picks.astype(int):
                f0 = frames[p]
                if f0["img_id"] not in track:
                    continue
                # partner frame with real camera travel
                j, travel = None, 0.0
                for q in range(p + 1, min(p + FLOW_MAX_GAP + 1, len(frames))):
                    d = float(np.linalg.norm(frames[q]["ee"] - f0["ee"]))
                    j, travel = q, d
                    if d >= FLOW_MIN_TRAVEL_M:
                        break
                if j is None or travel < FLOW_MIN_TRAVEL_M:
                    continue
                events.append({
                    "trial": label, "cycle": ci, "frame_idx": int(p),
                    "img_id": f0["img_id"], "depth_id": f0["depth_id"],
                    "partner_id": frames[j]["img_id"],
                    "gap": j - p, "travel": travel,
                    "width_mm": f0["width"] * 1000.0,
                    "rgb_dir": rgb_dir, "depth_dir": depth_dir,
                    "role": role, "gt_row": track[f0["img_id"]],
                })
    return events


def real_depth_m(path, H, W):
    if path.endswith(".npy"):
        d = np.load(path).astype(np.float32)
    else:
        d = np.array(Image.open(path)).astype(np.float32) / 1000.0
    if d.shape != (H, W):
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_cycle", type=int, default=3,
                    help="frames sampled from inside each grasp cycle")
    args = ap.parse_args()

    dev = pick_device()
    print(f"[setup] device={dev}", flush=True)
    print(f"[setup] sampling {args.per_cycle} frame(s) from inside each grasp "
          f"cycle, needing >= {FLOW_MIN_TRAVEL_M*1000:.0f} mm camera travel "
          f"and >= {MIN_BG_FLOW_PX} px background flow", flush=True)

    events = build_held_events(args)
    print(f"\n[events] {len(events)} mid-hold frames across "
          f"{len(set(e['trial'] for e in events))} trials", flush=True)
    if not events:
        print("[fatal] no events", flush=True)
        return

    print(f"[load] SAM {MODEL} from {CKPT}", flush=True)
    sam = sam_model_registry[MODEL](checkpoint=CKPT).to(dev)
    gen = SamAutomaticMaskGenerator(sam, points_per_side=16,
                                    pred_iou_thresh=0.86,
                                    stability_score_thresh=0.90)

    rows, panels = [], []
    for n, ev in enumerate(events):
        tag = f"{ev['trial']}/c{ev['cycle']}f{ev['frame_idx']}"
        rgb_p = os.path.join(ev["rgb_dir"], f"{ev['img_id']}.png")
        nxt_p = os.path.join(ev["rgb_dir"], f"{ev['partner_id']}.png")
        bgr, nxt = cv2.imread(rgb_p), cv2.imread(nxt_p)
        if bgr is None or nxt is None:
            print(f"[skip] {tag}: frame unreadable", flush=True)
            continue
        H, W = bgr.shape[:2]

        gt = mask_from_overlay(ev["gt_row"]["overlay_path"], rgb_p,
                               GRASPED_COLOR)
        if gt is None or gt.sum() == 0:
            print(f"[skip] {tag}: empty ground truth", flush=True)
            continue

        g0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 21, 3,
                                            5, 1.2, 0)
        fmag = np.linalg.norm(flow, axis=2)
        bg_med = float(np.median(fmag))
        if bg_med < MIN_BG_FLOW_PX:
            print(f"[skip] {tag}: background flow {bg_med:.2f} px, camera "
                  f"barely moved, event is undecidable", flush=True)
            continue

        # texture map: flow is unreliable where there is no gradient, which is
        # exactly how the previous rule got fooled
        gx = cv2.Sobel(g0, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g0, cv2.CV_32F, 0, 1, ksize=3)
        tex = np.abs(gx) + np.abs(gy)

        print(f"[run] {tag}: flow over {ev['gap']} frames / "
              f"{ev['travel']*1000:.1f} mm, bg {bg_med:.2f} px, "
              f"SAM proposals ...", flush=True)
        anns = gen.generate(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        border = np.zeros((H, W), dtype=bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True

        cands = []
        for a in anns:
            m = a["segmentation"]
            if m.sum() < MIN_AREA or (m & border).any():
                continue
            cands.append({
                "m": m,
                "flow": float(np.median(fmag[m])),
                "tex": float(np.mean(tex[m])),
                "area": int(m.sum()),
            })
        if len(cands) < 3:
            print(f"  [skip] only {len(cands)} proposals survived filtering",
                  flush=True)
            continue

        # adaptive texture gate: keep the better-textured half, so flow is
        # only trusted where it can be measured. No tuned constant.
        tex_thr = float(np.median([c["tex"] for c in cands]))
        textured = [c for c in cands if c["tex"] >= tex_thr] or cands

        pick = min(textured, key=lambda c: c["flow"])
        achieved = iou(pick["m"], gt)

        oracle = max(iou(c["m"], gt) for c in cands)
        rand = float(np.mean([iou(c["m"], gt) for c in cands]))
        largest = iou(max(cands, key=lambda c: c["area"])["m"], gt)
        cy, cx = H / 2.0, W / 2.0

        def centrality(c):
            ys, xs = np.nonzero(c["m"])
            return np.hypot(xs.mean() - cx, ys.mean() - cy)
        central = iou(min(cands, key=centrality)["m"], gt)

        depth_iou = float("nan")
        if ev["depth_dir"] and os.path.isdir(ev["depth_dir"]):
            dp = os.path.join(ev["depth_dir"], f"{ev['depth_id']}.png")
            if not os.path.exists(dp):
                dp = os.path.join(ev["depth_dir"], f"{ev['depth_id']}.npy")
            if os.path.exists(dp):
                d = real_depth_m(dp, H, W)
                ok = (d > 0.05) & (d < 5.0)
                best, bz = None, np.inf
                for c in cands:
                    sel = c["m"] & ok
                    if not sel.any():
                        continue
                    z = float(np.median(d[sel]))
                    if z < bz:
                        best, bz = c, z
                if best is not None:
                    depth_iou = iou(best["m"], gt)

        gt_flow = float(np.median(fmag[gt]))
        print(f"  [result] {tag}: stationarity IoU={achieved:.3f}  "
              f"random={rand:.3f}  depth={depth_iou:.3f}  "
              f"oracle={oracle:.3f}   gt flow={gt_flow:.2f} vs bg "
              f"{bg_med:.2f} px  ({len(textured)}/{len(cands)} textured)",
              flush=True)

        rows.append([ev["trial"], ev["cycle"], ev["frame_idx"], ev["img_id"],
                     f"{ev['width_mm']:.2f}", f"{ev['travel']:.4f}",
                     f"{achieved:.4f}", f"{rand:.4f}", f"{largest:.4f}",
                     f"{central:.4f}", f"{depth_iou:.4f}", f"{oracle:.4f}",
                     f"{pick['flow']:.3f}", f"{gt_flow:.3f}",
                     f"{bg_med:.3f}", len(textured), len(cands)])

        vis = bgr.copy()
        for m, col in ((gt, (0, 255, 0)), (pick["m"], (0, 0, 220))):
            cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cont, -1, col, 2)
        cv2.putText(vis, f"{ev['trial'][6:]} c{ev['cycle']} IoU={achieved:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2)
        panels.append(vis)

    if not rows:
        print("[fatal] nothing scored", flush=True)
        return

    def col(i):
        v = np.array([float(r[i]) for r in rows])
        return v[np.isfinite(v)]

    print(f"\n[summary] {len(rows)} mid-hold events, all baselines recomputed "
          f"on THIS set from the same proposal pool", flush=True)
    for name, i in [("stationarity", 6), ("random (null)", 7),
                    ("largest", 8), ("central", 9), ("depth", 10),
                    ("oracle (ceiling)", 11)]:
        v = col(i)
        if len(v):
            print(f"  {name:18s} n={len(v):3d}  mean IoU={v.mean():.3f}  "
                  f"median={np.median(v):.3f}", flush=True)

    st, rn = col(6), col(7)
    print(f"\n[verdict] stationarity {st.mean():.3f} vs random null "
          f"{rn.mean():.3f}", flush=True)
    if st.mean() > 2 * rn.mean() and st.mean() > 0.25:
        print("  BEATS the null clearly. The signal converts into a rule.",
              flush=True)
    elif st.mean() > rn.mean():
        print("  Beats the null but not decisively. Not enough to claim the "
              "selection problem is solved.", flush=True)
    else:
        print("  DOES NOT beat the null. The rule does not work, and the "
              "held-window ratio does not convert into proposal selection.",
              flush=True)
    print("  NOTE: this is a different event set from the 11-event one, so "
          "0.104 and 0.166\n  are NOT directly comparable. The recomputed "
          "rows above are the fair comparison.", flush=True)

    os.makedirs("figures", exist_ok=True)
    if os.path.exists(OUT_CSV):
        shutil.copy2(OUT_CSV, OUT_CSV + ".bak")
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "cycle", "frame_idx", "img_id", "width_mm",
                    "ee_travel_m", "iou_stationarity", "iou_random",
                    "iou_largest", "iou_central", "iou_depth", "iou_oracle",
                    "picked_flow_px", "gt_flow_px", "bg_flow_px",
                    "n_textured", "n_proposals"])
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
