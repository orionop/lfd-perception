"""
Build the Mark-facing JSON sidecar from artifacts we already have.

Reads:
    figures/propagation_summary.csv       (carrot, role=grasped)
    figures/propagation_cup_summary.csv   (cup,    role=contact_receiver)
    lfdws_t001/lfdws_t001/lfdws_t001_0.csv (events + image timestamps)
    figures/propagation/*.png             (carrot overlays)
    figures/propagation_cup/*.png         (cup overlays)

Writes:
    figures/identify/objects.json
    figures/identify/objects_summary.csv
    figures/identify/overlay.mp4   (per-frame combined overlay)

No model inference — pure aggregation. Use this when identify_objects.py
crashes due to memory pressure from running two SAM 2 objects together.
"""
import ast
import csv
import json
import os

import cv2
import numpy as np
import pandas as pd

DEMO_CSV = "Data/lfdws_t001/lfdws_t001/lfdws_t001_0.csv"
SRC_IMG_DIR = "Data/lfdws_t001/lfdws_t001/zed_zed_node_rgb_color_rect_image_compressed"
CARROT_CSV = "figures/propagation_summary.csv"
CUP_CSV = "figures/propagation_cup_summary.csv"
OUT_DIR = "figures/identify"

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
    df = pd.read_csv(csv_path)
    t = pd.to_datetime(df[POSE_TS])
    t_rel = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    w = df[GRIP].apply(parse_gw).to_numpy()
    fm = np.sqrt(df[FX].astype(float)**2 + df[FY].astype(float)**2 + df[FZ].astype(float)**2).to_numpy()
    w_open, w_closed = float(np.nanmax(w)), float(np.nanmin(w))
    thr = w_closed + 0.5 * (w_open - w_closed)
    closed = w < thr
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
    baseline = float(np.median(fm[: len(fm) // 10]))
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
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "overlays"), exist_ok=True)

    print(f"[load] events from {DEMO_CSV}", flush=True)
    events = events_from_demo(DEMO_CSV)
    for n, e in events.items():
        print(f"  {n:8s} t={e['t_rel_s']:6.2f}s img={e['img_ts']}", flush=True)

    # ---- pose timestamp -> t_rel ----
    df = pd.read_csv(DEMO_CSV)
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

    carrot_rows = load_csv(CARROT_CSV)
    cup_rows = load_csv(CUP_CSV)
    print(f"[load] carrot rows: {len(carrot_rows)}  cup rows: {len(cup_rows)}",
          flush=True)

    objects = {
        1: {"role": "grasped",          "color": [0, 255, 0],   "source_rows": carrot_rows},
        2: {"role": "contact_receiver", "color": [255, 0, 255], "source_rows": cup_rows},
    }

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
        "trial_dir": "Data/lfdws_t001/lfdws_t001",
        "csv": DEMO_CSV,
        "image_dir": SRC_IMG_DIR,
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
        for oid in (1, 2):
            if (oid, fidx) in lut:
                any_row = lut[(oid, fidx)]
                break
        if any_row is None:
            continue
        png = any_row["file"]
        src_path = os.path.join(SRC_IMG_DIR, png)
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
        ov_out = os.path.join(OUT_DIR, "overlays", f"f{fidx:04d}_{png}")
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
    json_path = os.path.join(OUT_DIR, "objects.json")
    with open(json_path, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"[write] {json_path}", flush=True)

    sum_path = os.path.join(OUT_DIR, "objects_summary.csv")
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
        mp4 = os.path.join(OUT_DIR, "overlay.mp4")
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
