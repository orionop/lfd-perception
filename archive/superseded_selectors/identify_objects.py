"""
End-to-end task-relevant object identification on any bag exported in the
lab's standard format.

Pipeline (per Vlutters, May 19 / 21):
    1. read merged CSV (lfdws_*_0.csv) — pose, gripper, wrench, image-ts
    2. detect proprioceptive events (grasp / press / release)
    3. seed SAM 2 at each event frame with a point prompt
    4. propagate masks across the demo
    5. write a JSON sidecar (object_id, role, per-frame masks, bbox, area)
    6. write per-frame overlay video + summary CSV

This is the Mark-facing entry point. Call:

    .venv_sam2/bin/python identify_objects.py --trial <bag_folder>

Assumes:
    - .venv_sam2 has SAM 2 installed
    - sam2.1_hiera_large.pt is in the cwd
    - frames have been jpg-converted into ./frames_jpg via prepare_sam2_frames.py
"""
import argparse
import ast
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from event_utils import gripper_moved
import pandas as pd
import torch
from sam2.build_sam import build_sam2_video_predictor

IMG_DIR_NAME = "zed_zed_node_rgb_color_rect_image_compressed"

# CSV columns (per Mark's spec, dot-separated, topic-prefixed)
POSE_TS = "NS_1.franka_robot_state_broadcaster.current_pose.timestamp"
POSE_X = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.x"
POSE_Y = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.y"
POSE_Z = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.z"
GRIP = "NS_1.franka_gripper.joint_states.position"
FX = "bota_post.wrench_body_compensated.wrench.force.x"
FY = "bota_post.wrench_body_compensated.wrench.force.y"
FZ = "bota_post.wrench_body_compensated.wrench.force.z"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"

# Seed point fractions (image-relative) per role — these are starting guesses.
# In a follow-up they would come from EE-projection; for now they're the
# fractions that worked on the carrot trial.
ROLE_SEEDS = {
    "grasped":          {"frac": (0.70, 0.30), "color": (0, 255, 0)},      # green
    "contact_receiver": {"frac": (0.55, 0.65), "color": (255, 0, 255)},    # magenta
}


def parse_gripper_width(cell):
    try:
        v = ast.literal_eval(cell)
        return float(np.sum(v))
    except Exception:
        return float("nan")


def load_demo(csv_path):
    df = pd.read_csv(csv_path)
    t = pd.to_datetime(df[POSE_TS])
    df["t_rel"] = (t - t.iloc[0]).dt.total_seconds()
    df["grip_w"] = df[GRIP].apply(parse_gripper_width)
    df["force_mag"] = np.sqrt(
        df[FX].astype(float) ** 2 + df[FY].astype(float) ** 2 + df[FZ].astype(float) ** 2
    )
    return df


def detect_events(df):
    """Return {event_name: (t_rel, row_idx, image_id_str)}."""
    events = {}
    w = df["grip_w"].to_numpy()
    w_open, w_closed = float(np.nanmax(w)), float(np.nanmin(w))
    thr = w_closed + 0.5 * (w_open - w_closed)
    # Guard: a gripper that never actuated puts this midpoint INSIDE the
    # sensor's own noise band and manufactures grasp/release out of nothing.
    # Measured on lfdws_t001_labexport: width spans 6.6e-7 m of pure noise yet
    # the unguarded rule reported a grasp at 0.06 s and a release at 7.66 s,
    # which then displaced the contact event from the true 11.15 N peak at
    # 5.08 s to 3.34 s. See Code/event_utils.py.
    closed = (w < thr) if gripper_moved(w) else np.zeros(len(w), dtype=bool)
    cd = np.where((~closed[:-1]) & (closed[1:]))[0] + 1
    cu = np.where((closed[:-1]) & (~closed[1:]))[0] + 1
    if len(cd):
        i = int(cd[0])
        events["grasp"] = (df["t_rel"].iloc[i], i, str(df[IMG].iloc[i]))
    if len(cu):
        i = int(cu[-1])
        events["release"] = (df["t_rel"].iloc[i], i, str(df[IMG].iloc[i]))

    fm = df["force_mag"].to_numpy()
    baseline = np.median(fm[: len(fm) // 10])
    fm_adj = np.where(closed, fm - baseline, -np.inf)
    i = int(np.argmax(fm_adj))
    events["press"] = (df["t_rel"].iloc[i], i, str(df[IMG].iloc[i]))
    return dict(sorted(events.items(), key=lambda kv: kv[1][0]))


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def overlay(img_bgr, mask, color, alpha=0.5):
    if mask is None or mask.sum() == 0:
        return img_bgr.copy()
    layer = np.zeros_like(img_bgr)
    layer[mask] = color
    return cv2.addWeighted(img_bgr, 1.0, layer, alpha, 0)


def bbox_of_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True,
                    help="bag folder containing the merged CSV + image subfolder")
    ap.add_argument("--jpg_dir", default="frames_jpg",
                    help="folder with 00000.jpg, ... (SAM 2 requirement)")
    ap.add_argument("--ckpt", default="sam2.1_hiera_large.pt")
    ap.add_argument("--cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--out", default="figures/identify")
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "overlays"), exist_ok=True)

    # ---------- locate inputs ----------
    csv_path = None
    for f in os.listdir(args.trial):
        if f.endswith(".csv") and not f.startswith("."):
            csv_path = os.path.join(args.trial, f)
            break
    if csv_path is None:
        print(f"[fatal] no merged CSV in {args.trial}", flush=True)
        sys.exit(1)
    img_dir = os.path.join(args.trial, IMG_DIR_NAME)
    print(f"[setup] CSV: {csv_path}", flush=True)
    print(f"[setup] images: {img_dir}", flush=True)

    df = load_demo(csv_path)
    print(f"[load] {len(df)} rows, {df['t_rel'].iloc[-1]:.1f}s demo", flush=True)
    events = detect_events(df)
    for n, (t, _, ts) in events.items():
        print(f"[event] {n:8s} t={t:6.2f}s  img={ts}", flush=True)

    png_frames = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    jpg_frames = sorted([f for f in os.listdir(args.jpg_dir) if f.endswith(".jpg")])
    assert len(jpg_frames) == len(png_frames), \
        f"jpg/png count mismatch: {len(jpg_frames)} vs {len(png_frames)}"

    probe = cv2.imread(os.path.join(img_dir, png_frames[0]))
    H, W = probe.shape[:2]

    # role assignment: gripper-close event -> grasped object; force-peak -> contact_receiver
    role_events = {
        "grasped":          events.get("grasp"),
        "contact_receiver": events.get("press"),
    }

    # ---------- SAM 2 init ----------
    device = pick_device()
    print(f"[load] SAM 2 ({device}) ...", flush=True)
    t0 = time.time()
    predictor = build_sam2_video_predictor(args.cfg, args.ckpt, device=device)
    print(f"[load] ready in {time.time()-t0:.1f}s", flush=True)

    print(f"[init] init_state on {args.jpg_dir}", flush=True)
    t0 = time.time()
    state = predictor.init_state(video_path=args.jpg_dir)
    print(f"[init] ok in {time.time()-t0:.1f}s", flush=True)

    # seed both objects
    obj_ids = {}
    next_obj = 1
    for role, ev in role_events.items():
        if ev is None:
            continue
        _, row_idx, img_ts = ev
        seed_name = f"{img_ts}.png"
        if seed_name not in png_frames:
            print(f"[warn] seed frame {seed_name} for {role} not found", flush=True)
            continue
        seed_idx = png_frames.index(seed_name)
        fx, fy = ROLE_SEEDS[role]["frac"]
        px, py = fx * W, fy * H
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=seed_idx,
            obj_id=next_obj,
            points=np.array([[px, py]], dtype=np.float32),
            labels=np.array([1], dtype=np.int32),
        )
        obj_ids[next_obj] = {"role": role, "seed_frame_idx": seed_idx,
                             "seed_point": [px, py], "seed_img_ts": img_ts,
                             "color": ROLE_SEEDS[role]["color"]}
        print(f"[seed] obj_id={next_obj} role={role} frame={seed_idx} pt=({px:.0f},{py:.0f})",
              flush=True)
        next_obj += 1

    # ---------- propagate both directions from earliest seed ----------
    earliest_seed = min(o["seed_frame_idx"] for o in obj_ids.values())
    latest_seed = max(o["seed_frame_idx"] for o in obj_ids.values())
    print(f"[propagate] forward from {earliest_seed} and backward from {latest_seed}",
          flush=True)

    # per-frame, per-obj mask aggregation
    masks_by_frame = {}  # frame_idx -> { obj_id: mask_bool }

    def collect(out_frame_idx, out_obj_ids, out_mask_logits):
        if out_frame_idx not in masks_by_frame:
            masks_by_frame[out_frame_idx] = {}
        for oid, logits in zip(out_obj_ids, out_mask_logits):
            m = (logits > 0.0).cpu().numpy().squeeze().astype(bool)
            masks_by_frame[out_frame_idx][int(oid)] = m

    n_done = 0
    t0 = time.time()
    for fidx, oids, logits in predictor.propagate_in_video(state, start_frame_idx=earliest_seed):
        collect(fidx, oids, logits)
        n_done += 1
        if n_done % 25 == 0 or n_done == 1:
            print(f"  [fwd] f{fidx:3d}  objs={list(masks_by_frame[fidx].keys())}  "
                  f"({n_done} done, {n_done/(time.time()-t0):.2f} f/s)", flush=True)

    n_done = 0
    t0 = time.time()
    for fidx, oids, logits in predictor.propagate_in_video(
            state, start_frame_idx=latest_seed, reverse=True):
        if fidx in masks_by_frame and all(oid in masks_by_frame[fidx] for oid in obj_ids):
            continue  # already covered by forward pass
        collect(fidx, oids, logits)
        n_done += 1
        if n_done % 25 == 0 or n_done == 1:
            print(f"  [bwd] f{fidx:3d}  objs={list(masks_by_frame[fidx].keys())}  "
                  f"({n_done} done, {n_done/(time.time()-t0):.2f} f/s)", flush=True)

    # ---------- per-frame outputs ----------
    print("[write] per-frame overlays + JSON sidecar", flush=True)
    sidecar = {
        "trial_dir": args.trial,
        "csv": csv_path,
        "image_dir": img_dir,
        "events": {n: {"t_rel_s": float(t), "row_idx": int(ri), "img_ts": ts}
                   for n, (t, ri, ts) in events.items()},
        "objects": {str(oid): {**info, "color": list(info["color"])}
                    for oid, info in obj_ids.items()},
        "frames": [],
    }

    for fidx in sorted(masks_by_frame.keys()):
        png = png_frames[fidx]
        bgr = cv2.imread(os.path.join(img_dir, png))
        comp = bgr.copy()
        per_obj = []
        for oid, m in masks_by_frame[fidx].items():
            info = obj_ids[oid]
            comp = overlay(comp, m, info["color"])
            bb = bbox_of_mask(m)
            per_obj.append({
                "obj_id": oid, "role": info["role"],
                "mask_px": int(m.sum()),
                "bbox_xyxy": bb,
            })
        cv2.putText(comp, f"f{fidx:03d}  {png}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        ov_path = os.path.join(args.out, "overlays", f"f{fidx:04d}_{png}")
        cv2.imwrite(ov_path, comp)
        sidecar["frames"].append({
            "frame_idx": fidx,
            "img_filename": png,
            "overlay_path": ov_path,
            "objects": per_obj,
        })

    sidecar_path = os.path.join(args.out, "objects.json")
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[write] sidecar -> {sidecar_path}", flush=True)

    # per-frame summary CSV
    sum_csv = os.path.join(args.out, "objects_summary.csv")
    with open(sum_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "img_filename", "obj_id", "role", "mask_px",
                    "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "overlay_path"])
        for fr in sidecar["frames"]:
            for o in fr["objects"]:
                bb = o["bbox_xyxy"] or [-1, -1, -1, -1]
                w.writerow([fr["frame_idx"], fr["img_filename"], o["obj_id"],
                            o["role"], o["mask_px"], *bb, fr["overlay_path"]])
    print(f"[write] summary csv -> {sum_csv}", flush=True)

    # stitch MP4
    if sidecar["frames"]:
        sample = cv2.imread(sidecar["frames"][0]["overlay_path"])
        h, w = sample.shape[:2]
        mp4 = os.path.join(args.out, "overlay.mp4")
        vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
        print(f"[video] writing {mp4} ({len(sidecar['frames'])} frames @ 15 fps)",
              flush=True)
        for i, fr in enumerate(sidecar["frames"]):
            vw.write(cv2.imread(fr["overlay_path"]))
            if (i + 1) % 100 == 0:
                print(f"  [video] {i+1}/{len(sidecar['frames'])}", flush=True)
        vw.release()
        print(f"[video] saved -> {mp4}", flush=True)

    print("[done]", flush=True)


if __name__ == "__main__":
    main()
