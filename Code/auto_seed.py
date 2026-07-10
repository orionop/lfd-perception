"""
Auto-seed point picker for SAM 2 propagation.

Without camera extrinsics, we cannot project the end-effector pose into the
image plane. Until that's available, we use a vision-only heuristic:
  - run SAM's automatic mask generator on the event frame
  - score each candidate mask by (centrality, plausible size, distance from
    image border)
  - return the centroid of the top-scoring mask as the seed point

This replaces the hand-tuned `SEED_POINT_FRAC` constants in
propagate_demo.py / propagate_cup.py. The seed CSV it writes is consumed
downstream by `run_pipeline.py`.

Usage:
    .venv_sam2/bin/python Code/auto_seed.py --trial Data/lfdws_t001/lfdws_t001 \
        --ckpt sam_vit_h_4b8939.pth
"""
import argparse
import ast
import csv
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

IMG_DIR_NAME = "zed_zed_node_rgb_color_rect_image_compressed"

POSE_TS = "NS_1.franka_robot_state_broadcaster.current_pose.timestamp"
GRIP = "NS_1.franka_gripper.joint_states.position"
FX = "bota_post.wrench_body_compensated.wrench.force.x"
FY = "bota_post.wrench_body_compensated.wrench.force.y"
FZ = "bota_post.wrench_body_compensated.wrench.force.z"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"


def pick_device():
    # SamAutomaticMaskGenerator builds float64 point grids internally,
    # which MPS rejects. CPU is the safe path (only a handful of frames here).
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_gw(c):
    try:
        return float(np.sum(ast.literal_eval(c)))
    except Exception:
        return float("nan")


def detect_events(df):
    """Same logic as analyze_demo.py — returns {name: (t_rel, idx, img_id)}.

    If the trial has no gripper topic (see Docs/FAILURE_MODES.md B3), skip
    grasp/release entirely and pick 'press' as the single strongest
    force-magnitude peak over the WHOLE trace (no held-window restriction,
    since there's no grasp/release to bound it).

    If the trial has no wrench topic (symmetric gap -- e.g. bota
    disconnected; confirmed on lfdws_t004/lfdws_t005), skip 'press'
    entirely and return grasp/release only. The 'contact_receiver' role
    then has nothing to seed from, same as 'grasped' being skipped when
    there's no grasp event."""
    t = pd.to_datetime(df[POSE_TS])
    t_rel = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    has_force = FX in df.columns
    if has_force:
        fm = np.sqrt(df[FX].astype(float) ** 2 + df[FY].astype(float) ** 2 +
                     df[FZ].astype(float) ** 2).to_numpy()
        baseline = float(np.median(fm[: len(fm) // 10]))

    if GRIP not in df.columns:
        print("[detect] no gripper topic -- force-only fallback "
              "(grasp/release skipped)", flush=True)
        if not has_force:
            print("[detect] no gripper AND no wrench topic -- no events "
                  "detectable", flush=True)
            return {}
        fm_adj = fm - baseline
        i = int(np.argmax(fm_adj))
        return {"press": (float(t_rel[i]), i, str(df[IMG].iloc[i]))}

    w = df[GRIP].apply(parse_gw).to_numpy()
    w_open, w_closed = float(np.nanmax(w)), float(np.nanmin(w))
    thr = w_closed + 0.5 * (w_open - w_closed)
    closed = w < thr
    cd = np.where((~closed[:-1]) & (closed[1:]))[0] + 1
    cu = np.where((closed[:-1]) & (~closed[1:]))[0] + 1
    out = {}
    if len(cd):
        i = int(cd[0]); out["grasp"] = (float(t_rel[i]), i, str(df[IMG].iloc[i]))
    if len(cu):
        i = int(cu[-1]); out["release"] = (float(t_rel[i]), i, str(df[IMG].iloc[i]))
    if not has_force:
        print("[detect] no wrench topic -- gripper-only fallback "
              "(press skipped)", flush=True)
        return out
    fm_adj = np.where(closed, fm - baseline, -np.inf)
    i = int(np.argmax(fm_adj))
    out["press"] = (float(t_rel[i]), i, str(df[IMG].iloc[i]))
    return out


def score_mask(m, img_shape, role):
    """Higher = better candidate seed.

    For 'grasped' (carrot at grasp): prefer masks in the upper part of the
    frame (where the gripper enters) with medium area.
    For 'contact_receiver' (cup at press): prefer masks in the lower-centre
    with larger area.

    NOTE: the absolute area cutoff below (af > 0.4 rejected) is tuned to
    lfdws_t001's object scale and known to misfire on scenes with a larger
    contact-receiver object (see Docs/FAILURE_MODES.md B4 -- the plate on
    lfdws_t001_depth was wrongly rejected by this cutoff). A relative-size
    fix (reward largest-of-candidates instead of an absolute fraction) was
    tried and REVERTED: on lfdws_t001 it just as reliably mis-picks the
    reflective purple table mat as the "largest candidate" instead of the
    cup (verified -- see figures/identify/auto_seeds_VERIFY_generalized.png,
    kept as a negative-result artifact). Neither an absolute nor a relative
    area heuristic is safe here; SAM-only seed picking is fundamentally
    unreliable across differently-scaled/textured scenes. The real fix is
    project_ee.py's geometric EE-projection seed, blocked on calibration.
    """
    H, W = img_shape[:2]
    seg = m["segmentation"]
    area = int(seg.sum())
    ys, xs = np.where(seg)
    if len(xs) == 0:
        return -np.inf, None
    cx, cy = xs.mean(), ys.mean()

    # reject masks touching the image border on more than two sides (likely background)
    border = ((seg[0, :].any()) + (seg[-1, :].any()) +
              (seg[:, 0].any()) + (seg[:, -1].any()))
    if border >= 3:
        return -np.inf, (cx, cy)

    # area sanity
    img_area = H * W
    af = area / img_area
    if af < 0.005 or af > 0.4:
        return -np.inf, (cx, cy)

    # centrality term (horizontal): prefer near image centre
    horiz = 1.0 - abs(cx - W / 2) / (W / 2)

    if role == "grasped":
        # gripper enters from the top of the frame in lfdws_t001; reward upper half
        vert = 1.0 - cy / H
        size = 1.0 - abs(af - 0.05) * 10  # ~5% of image is a typical small held object
    else:  # contact_receiver
        vert = cy / H  # reward lower half
        size = 1.0 - abs(af - 0.10) * 5   # larger target object

    return 0.5 * horiz + 0.3 * vert + 0.2 * max(0.0, size), (cx, cy)


def pick_seed(img_bgr, masks, role):
    best, best_pt, best_mask = -np.inf, None, None
    for m in masks:
        s, pt = score_mask(m, img_bgr.shape, role)
        if s > best:
            best, best_pt, best_mask = s, pt, m
    return best, best_pt, best_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--ckpt", default="sam_vit_h_4b8939.pth")
    ap.add_argument("--model", default="vit_h")
    ap.add_argument("--out_csv", default="figures/identify/auto_seeds.csv")
    ap.add_argument("--out_overlay", default="figures/identify/auto_seeds.png")
    ap.add_argument("--points_per_side", type=int, default=24)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    csv_path = next((os.path.join(args.trial, f) for f in os.listdir(args.trial)
                     if f.endswith(".csv") and not f.startswith(".")), None)
    if csv_path is None:
        print(f"[fatal] no merged CSV in {args.trial}", flush=True); sys.exit(1)
    img_dir = os.path.join(args.trial, IMG_DIR_NAME)

    df = pd.read_csv(csv_path)
    events = detect_events(df)
    role_for_event = {"grasp": "grasped", "press": "contact_receiver"}
    print(f"[load] CSV={csv_path}", flush=True)
    for n, (t, _, ts) in events.items():
        print(f"[event] {n:8s} t={t:6.2f}s img={ts}", flush=True)

    device = pick_device()
    print(f"[load] SAM ({args.model}) on {device}", flush=True)
    sam = sam_model_registry[args.model](checkpoint=args.ckpt).to(device)
    mg = SamAutomaticMaskGenerator(
        sam, points_per_side=args.points_per_side,
        pred_iou_thresh=0.85, stability_score_thresh=0.9,
        min_mask_region_area=400,
    )

    rows = [("role", "event", "img_id", "seed_x", "seed_y",
             "frac_x", "frac_y", "mask_px")]
    panels = []
    for event_name, role in role_for_event.items():
        ev = events.get(event_name)
        if ev is None:
            print(f"[skip] {event_name}: no event detected", flush=True); continue
        _, _, img_id = ev
        path = os.path.join(img_dir, f"{img_id}.png")
        bgr = cv2.imread(path)
        if bgr is None:
            print(f"[skip] {event_name}: missing {path}", flush=True); continue
        H, W = bgr.shape[:2]
        print(f"[gen]  {event_name}: SAM auto-mask ({W}x{H}) ...", flush=True)
        masks = mg.generate(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        print(f"[gen]  {event_name}: {len(masks)} candidate masks", flush=True)

        score, pt, best_mask = pick_seed(bgr, masks, role)
        if pt is None:
            print(f"[warn] {event_name}: no candidate passed scoring", flush=True)
            continue
        cx, cy = pt
        fx, fy = cx / W, cy / H
        mask_px = int(best_mask["segmentation"].sum()) if best_mask is not None else 0
        rows.append((role, event_name, img_id, f"{cx:.1f}", f"{cy:.1f}",
                     f"{fx:.4f}", f"{fy:.4f}", mask_px))
        print(f"[pick] {event_name} -> role={role} seed=({cx:.0f},{cy:.0f})"
              f" frac=({fx:.3f},{fy:.3f}) score={score:.2f} mask_px={mask_px}",
              flush=True)

        ov = bgr.copy()
        if best_mask is not None:
            color = (0, 255, 0) if role == "grasped" else (255, 0, 255)
            layer = np.zeros_like(ov); layer[best_mask["segmentation"]] = color
            ov = cv2.addWeighted(ov, 1.0, layer, 0.4, 0)
        cv2.circle(ov, (int(cx), int(cy)), 10, (0, 0, 255), -1)
        cv2.circle(ov, (int(cx), int(cy)), 10, (255, 255, 255), 2)
        cv2.putText(ov, f"{event_name} -> {role}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        panels.append(ov)

    with open(args.out_csv, "w") as f:
        csv.writer(f).writerows(rows)
    print(f"[write] {args.out_csv}", flush=True)

    if panels:
        target_h = 480
        resized = [cv2.resize(p, (int(p.shape[1] * target_h / p.shape[0]), target_h))
                   for p in panels]
        cv2.imwrite(args.out_overlay, np.hstack(resized))
        print(f"[write] {args.out_overlay}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
