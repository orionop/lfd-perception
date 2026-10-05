"""
Build the Mark-facing JSON sidecar from artifacts we already have.

Reads (paths derived from --trial, --carrot_csv, --cup_csv):
    <carrot_csv>       propagate_demo.py summary (carrot, role=grasped)
    <cup_csv>          propagate_cup.py summary  (cup,    role=contact_receiver)
    <trial>/<trial>_0.csv                        (events + image timestamps)
    figures/propagation/*.png                    (carrot overlays)
    figures/propagation_cup*/*.png                (cup overlays)

Writes (under --out):
    objects.json
    objects_summary.csv
    overlay.mp4   (per-frame combined overlay)

No model inference — pure aggregation. Use this when identify_objects.py
crashes due to memory pressure from running two SAM 2 objects together.

Handles trials with only ONE propagated object (e.g. lfdws_t001_depth has
no gripper topic, so there's no grasp event / no propagate_demo.py run —
only the contact_receiver from propagate_cup.py). Whichever of
--carrot_csv / --cup_csv is missing or empty is simply omitted from
"objects" rather than erroring.

Usage:
    .venv_analysis/bin/python Code/build_sidecar.py \
        --trial Data/lfdws_t001/lfdws_t001 \
        --carrot_csv figures/propagation_summary.csv \
        --cup_csv figures/propagation_cup_summary.csv \
        --out figures/identify
"""
import argparse
import ast
import csv
import json
import os
import shutil

import cv2
import numpy as np

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from event_utils import gripper_moved
import pandas as pd

IMG_DIR_NAME = "zed_zed_node_rgb_color_rect_image_compressed"


def backup_if_exists(path):
    """Before overwriting a deliverable, save whatever was already there
    to path + '.bak'. Same insurance as propagate_cup.py/propagate_demo.py
    (see Docs/FAILURE_MODES.md C1b)."""
    if os.path.exists(path):
        bak = path + ".bak"
        shutil.copy2(path, bak)
        print(f"[backup] {path} -> {bak}", flush=True)

POSE_TS = "NS_1.franka_robot_state_broadcaster.current_pose.timestamp"
GRIP = "NS_1.franka_gripper.joint_states.position"
FX = "bota_post.wrench_body_compensated.wrench.force.x"
FY = "bota_post.wrench_body_compensated.wrench.force.y"
FZ = "bota_post.wrench_body_compensated.wrench.force.z"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"


def parse_gw(c):
    try:
        return float(np.sum(ast.literal_eval(c)))
    except Exception:
        return float("nan")


def events_from_demo(csv_path):
    """Same force-only fallback as auto_seed.py / force_only_events.py:
    if there's no gripper topic, skip grasp/release and report the single
    strongest force-magnitude peak over the whole trace as 'press'.

    Symmetric case (confirmed on lfdws_t004/lfdws_t005): no wrench topic
    at all -- skip 'press' entirely and return grasp/release only."""
    df = pd.read_csv(csv_path)
    t = pd.to_datetime(df[POSE_TS])
    t_rel = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    has_force = FX in df.columns
    if has_force:
        fm = np.sqrt(df[FX].astype(float)**2 + df[FY].astype(float)**2 + df[FZ].astype(float)**2).to_numpy()
        baseline = float(np.median(fm[: len(fm) // 10]))

    if GRIP not in df.columns:
        print("[events] no gripper topic -- force-only fallback", flush=True)
        if not has_force:
            print("[events] no gripper AND no wrench topic -- no events "
                  "detectable", flush=True)
            return {}
        fm_adj = fm - baseline
        i = int(np.argmax(fm_adj))
        return {"press": {"t_rel_s": float(t_rel[i]), "row_idx": i,
                          "img_ts": str(df[IMG].iloc[i])}}

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
    cd = np.where((~closed[:-1]) & (closed[1:]))[0] + 1
    cu = np.where((closed[:-1]) & (~closed[1:]))[0] + 1
    out = {}
    if len(cd):
        i = int(cd[0])
        out["grasp"] = {"t_rel_s": float(t_rel[i]), "row_idx": i,
                        "img_ts": str(df[IMG].iloc[i])}
    if len(cu):
        i = int(cu[-1])
        out["release"] = {"t_rel_s": float(t_rel[i]), "row_idx": i,
                          "img_ts": str(df[IMG].iloc[i])}
    if not has_force:
        print("[events] no wrench topic -- gripper-only fallback "
              "(press skipped)", flush=True)
        return dict(sorted(out.items(), key=lambda kv: kv[1]["t_rel_s"]))
    fm_adj = np.where(closed, fm - baseline, -np.inf)
    i = int(np.argmax(fm_adj))
    out["press"] = {"t_rel_s": float(t_rel[i]), "row_idx": i,
                    "img_ts": str(df[IMG].iloc[i])}
    return dict(sorted(out.items(), key=lambda kv: kv[1]["t_rel_s"]))


def mask_from_overlay(ov_path, src_path, color):
    ov = cv2.imread(ov_path)
    src = cv2.imread(src_path)
    if ov is None or src is None or ov.shape != src.shape:
        return None
    diff = ov.astype(int) - src.astype(int)
    if color == "green":
        return diff[..., 1] > 40
    if color == "magenta":
        return (diff[..., 0] > 40) & (diff[..., 2] > 40)
    return None


def bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", default="Data/lfdws_t001/lfdws_t001")
    ap.add_argument("--carrot_csv", default="figures/propagation_summary.csv")
    ap.add_argument("--cup_csv", default="figures/propagation_cup_summary.csv")
    ap.add_argument("--out", default="figures/identify")
    args = ap.parse_args()

    demo_csv = next((os.path.join(args.trial, f) for f in os.listdir(args.trial)
                     if f.endswith(".csv") and not f.startswith(".")), None)
    if demo_csv is None:
        raise FileNotFoundError(f"no merged CSV in {args.trial}")
    src_img_dir = os.path.join(args.trial, IMG_DIR_NAME)
    out_dir = args.out

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "overlays"), exist_ok=True)

    print(f"[load] events from {demo_csv}", flush=True)
    events = events_from_demo(demo_csv)
    for n, e in events.items():
        print(f"  {n:8s} t={e['t_rel_s']:6.2f}s img={e['img_ts']}", flush=True)

    # ---- pose timestamp -> t_rel ----
    df = pd.read_csv(demo_csv)
    t = pd.to_datetime(df[POSE_TS])
    trel = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    imgs = df[IMG].astype(str).to_numpy()
    img_to_trel = {}
    for tr, im in zip(trel, imgs):
        if im not in img_to_trel:
            img_to_trel[im] = float(tr)

    # ---- load propagation summaries ----
    def load_csv(path):
        rows = []
        if not os.path.exists(path):
            return rows
        with open(path) as f:
            for r in csv.DictReader(f):
                rows.append(r)
        return rows

    carrot_rows = load_csv(args.carrot_csv)
    cup_rows = load_csv(args.cup_csv)
    print(f"[load] carrot rows: {len(carrot_rows)}  cup rows: {len(cup_rows)}",
          flush=True)

    all_objects = {
        1: {"role": "grasped",          "color": [0, 255, 0],   "source_rows": carrot_rows},
        2: {"role": "contact_receiver", "color": [255, 0, 255], "source_rows": cup_rows},
    }
    objects = {oid: info for oid, info in all_objects.items() if info["source_rows"]}
    if len(objects) < len(all_objects):
        missing = [info["role"] for info in all_objects.values() if not info["source_rows"]]
        print(f"[info] no data for role(s) {missing} -- omitting from sidecar "
              f"(expected for trials without a grasp event, e.g. no gripper topic)",
              flush=True)

    # ---- per-frame aggregate ----
    # collect all unique frame_idx across both objects
    frame_set = set()
    for oid, info in objects.items():
        for r in info["source_rows"]:
            frame_set.add(int(r["frame_idx"]))
    frame_list = sorted(frame_set)
    print(f"[agg] {len(frame_list)} distinct frames across both objects", flush=True)

    # quick lookup: (obj_id, frame_idx) -> row
    lut = {}
    for oid, info in objects.items():
        for r in info["source_rows"]:
            lut[(oid, int(r["frame_idx"]))] = r

    sidecar = {
        "trial_dir": args.trial,
        "csv": demo_csv,
        "image_dir": src_img_dir,
        "events": events,
        "objects": {str(oid): {"role": info["role"], "color": info["color"]}
                    for oid, info in objects.items()},
        "frames": [],
    }
    summary_rows = []
    overlay_paths = []

    # one png per frame: base img with both masks overlaid
    print("[render] building combined overlays + per-frame records", flush=True)
    for n_done, fidx in enumerate(frame_list):
        # pick a source filename from whichever object has this frame
        any_row = None
        for oid in objects:
            if (oid, fidx) in lut:
                any_row = lut[(oid, fidx)]
                break
        if any_row is None:
            continue
        png = any_row["file"]
        src_path = os.path.join(src_img_dir, png)
        src = cv2.imread(src_path)
        if src is None:
            continue
        comp = src.copy()
        per_obj = []
        for oid, info in objects.items():
            if (oid, fidx) not in lut:
                continue
            ov_path_single = lut[(oid, fidx)]["overlay_path"]
            color_name = "green" if oid == 1 else "magenta"
            m = mask_from_overlay(ov_path_single, src_path, color_name)
            if m is None or m.sum() == 0:
                continue
            layer = np.zeros_like(src)
            layer[m] = info["color"]
            comp = cv2.addWeighted(comp, 1.0, layer, 0.5, 0)
            bb = bbox(m)
            per_obj.append({
                "obj_id": oid, "role": info["role"],
                "mask_px": int(m.sum()),
                "bbox_xyxy": bb,
            })
        cv2.putText(comp, f"f{fidx:03d}  {png}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        ov_out = os.path.join(out_dir, "overlays", f"f{fidx:04d}_{png}")
        cv2.imwrite(ov_out, comp)
        overlay_paths.append(ov_out)

        img_id_str = png.replace(".png", "")
        sidecar["frames"].append({
            "frame_idx": fidx,
            "img_filename": png,
            "t_rel_s": img_to_trel.get(img_id_str),
            "overlay_path": ov_out,
            "objects": per_obj,
        })
        for o in per_obj:
            bb = o["bbox_xyxy"] or [-1, -1, -1, -1]
            summary_rows.append([fidx, png, o["obj_id"], o["role"], o["mask_px"],
                                 *bb, ov_out])
        if (n_done + 1) % 50 == 0 or n_done == 0:
            print(f"  [render] {n_done+1}/{len(frame_list)} frames composed",
                  flush=True)

    # ---- write sidecar JSON ----
    json_path = os.path.join(out_dir, "objects.json")
    backup_if_exists(json_path)
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[write] {json_path}", flush=True)

    sum_path = os.path.join(out_dir, "objects_summary.csv")
    backup_if_exists(sum_path)
    with open(sum_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "img_filename", "obj_id", "role", "mask_px",
                    "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "overlay_path"])
        for row in summary_rows:
            w.writerow(row)
    print(f"[write] {sum_path}  ({len(summary_rows)} rows)", flush=True)

    # ---- MP4 ----
    if overlay_paths:
        sample = cv2.imread(overlay_paths[0])
        h, w = sample.shape[:2]
        mp4 = os.path.join(out_dir, "overlay.mp4")
        backup_if_exists(mp4)
        vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
        print(f"[video] writing {mp4} ({len(overlay_paths)} frames @ 15 fps)",
              flush=True)
        for i, p in enumerate(overlay_paths):
            vw.write(cv2.imread(p))
            if (i + 1) % 100 == 0:
                print(f"  [video] {i+1}/{len(overlay_paths)}", flush=True)
        vw.release()
        print(f"[video] {mp4}", flush=True)

    print("[done]", flush=True)


if __name__ == "__main__":
    main()
