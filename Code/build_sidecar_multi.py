"""
CANONICAL sidecar builder (2026-07-10): supersedes build_sidecar.py's
fixed-two-role {1: grasped, 2: contact_receiver} scheme, which cannot
represent trials with more than two task-relevant objects (confirmed on
lfdws_t001_depth: plate, screwdriver, charger, docking target -- a real
>=4-object task the fixed-pair schema can't hold). build_sidecar.py is
kept for reference/backward compat but is no longer the one to reach for
on new trials -- use this script.

Takes an arbitrary list of (obj_id, role, summary_csv, color) specs on
the command line instead of a hardcoded 2-slot dict, and composes the
same JSON sidecar shape (events + per-frame per-object records) for
however many objects are supplied, N=1 included.

Regression-verified against build_sidecar.py on lfdws_t001 (the original
2-role trial): same row count (737), and where the two differ, this
script is the more correct one -- its mask_from_overlay checks all 3
BGR channels against the target color, where build_sidecar.py's legacy
"green"/"magenta" check only tests 1-2 channels and can pick up
anti-aliasing noise at mask edges as false-positive pixels (confirmed
visually on lfdws_t001 frame 257: legacy mask included scattered noisy
edge pixels the stricter 3-channel check correctly excludes, 21565px vs
18517px for the same true mask).

Same event detection as build_sidecar.py (force-only fallback when no
gripper topic). Same per-frame combined-overlay + MP4 assembly.

Usage:
    .venv_analysis/bin/python Code/build_sidecar_multi.py \
        --trial Data/lfdws_t001_depth \
        --object 2:contact_receiver:figures/propagation_plate_depth_summary.csv:255,0,255 \
        --object 3:tool_contact:figures/propagation_obj3_screwdriver_summary.csv:0,165,255 \
        --out figures/identify_depth_multi
"""
import argparse
import ast
import csv
import json
import os
import shutil

import cv2
import numpy as np
import pandas as pd

IMG_DIR_NAME = "zed_zed_node_rgb_color_rect_image_compressed"

POSE_TS = "NS_1.franka_robot_state_broadcaster.current_pose.timestamp"
GRIP = "NS_1.franka_gripper.joint_states.position"
FX = "bota_post.wrench_body_compensated.wrench.force.x"
FY = "bota_post.wrench_body_compensated.wrench.force.y"
FZ = "bota_post.wrench_body_compensated.wrench.force.z"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"


def backup_if_exists(path):
    if os.path.exists(path):
        bak = path + ".bak"
        shutil.copy2(path, bak)
        print(f"[backup] {path} -> {bak}", flush=True)


def parse_gw(c):
    try:
        return float(np.sum(ast.literal_eval(c)))
    except Exception:
        return float("nan")


def events_from_demo(csv_path):
    """Same force-only fallback as build_sidecar.py."""
    df = pd.read_csv(csv_path)
    t = pd.to_datetime(df[POSE_TS])
    t_rel = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    fm = np.sqrt(df[FX].astype(float)**2 + df[FY].astype(float)**2 + df[FZ].astype(float)**2).to_numpy()
    baseline = float(np.median(fm[: len(fm) // 10]))
    if GRIP not in df.columns:
        fm_adj = fm - baseline
        i = int(np.argmax(fm_adj))
        return {"press": {"t_rel_s": float(t_rel[i]), "row_idx": i,
                          "img_ts": str(df[IMG].iloc[i])}}
    w = df[GRIP].apply(parse_gw).to_numpy()
    w_open, w_closed = float(np.nanmax(w)), float(np.nanmin(w))
    thr = w_closed + 0.5 * (w_open - w_closed)
    closed = w < thr
    cd = np.where((~closed[:-1]) & (closed[1:]))[0] + 1
    cu = np.where((closed[:-1]) & (~closed[1:]))[0] + 1
    out = {}
    if len(cd):
        i = int(cd[0]); out["grasp"] = {"t_rel_s": float(t_rel[i]), "row_idx": i, "img_ts": str(df[IMG].iloc[i])}
    if len(cu):
        i = int(cu[-1]); out["release"] = {"t_rel_s": float(t_rel[i]), "row_idx": i, "img_ts": str(df[IMG].iloc[i])}
    fm_adj = np.where(closed, fm - baseline, -np.inf)
    i = int(np.argmax(fm_adj))
    out["press"] = {"t_rel_s": float(t_rel[i]), "row_idx": i, "img_ts": str(df[IMG].iloc[i])}
    return dict(sorted(out.items(), key=lambda kv: kv[1]["t_rel_s"]))


def mask_from_overlay(ov_path, src_path, color_bgr, tol=40):
    ov = cv2.imread(ov_path)
    src = cv2.imread(src_path)
    if ov is None or src is None or ov.shape != src.shape:
        return None
    diff = ov.astype(int) - src.astype(int)
    b, g, r = color_bgr
    m = np.ones(diff.shape[:2], dtype=bool)
    for ch, target in zip(range(3), (b, g, r)):
        if target > 100:
            m &= diff[..., ch] > tol
        else:
            m &= diff[..., ch] < tol
    return m


def bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def parse_object_spec(spec):
    """obj_id:role:summary_csv:b,g,r"""
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(f"bad --object spec: {spec!r} (want obj_id:role:csv:b,g,r)")
    obj_id, role, csv_path, color_str = parts
    color = tuple(int(c) for c in color_str.split(","))
    return int(obj_id), role, csv_path, color


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--object", action="append", required=True,
                    help="obj_id:role:summary_csv:b,g,r -- repeatable, one per object")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    demo_csv = next((os.path.join(args.trial, f) for f in os.listdir(args.trial)
                     if f.endswith(".csv") and not f.startswith(".")), None)
    if demo_csv is None:
        raise FileNotFoundError(f"no merged CSV in {args.trial}")
    src_img_dir = os.path.join(args.trial, IMG_DIR_NAME)
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "overlays"), exist_ok=True)

    specs = [parse_object_spec(s) for s in args.object]
    print(f"[setup] {len(specs)} object(s): "
          f"{[(oid, role) for oid, role, _, _ in specs]}", flush=True)

    print(f"[load] events from {demo_csv}", flush=True)
    events = events_from_demo(demo_csv)
    for n, e in events.items():
        print(f"  {n:8s} t={e['t_rel_s']:6.2f}s img={e['img_ts']}", flush=True)

    df = pd.read_csv(demo_csv)
    t = pd.to_datetime(df[POSE_TS])
    trel = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    imgs = df[IMG].astype(str).to_numpy()
    img_to_trel = {}
    for tr, im in zip(trel, imgs):
        if im not in img_to_trel:
            img_to_trel[im] = float(tr)

    def load_csv(path):
        rows = []
        if not os.path.exists(path):
            print(f"  [warn] {path} missing -- skipping this object", flush=True)
            return rows
        with open(path) as f:
            for r in csv.DictReader(f):
                rows.append(r)
        return rows

    objects = {}
    for obj_id, role, csv_path, color in specs:
        rows = load_csv(csv_path)
        print(f"  [load] obj_id={obj_id} role={role}: {len(rows)} rows from {csv_path}",
              flush=True)
        if rows:
            objects[obj_id] = {"role": role, "color": color, "source_rows": rows}

    frame_set = set()
    for oid, info in objects.items():
        for r in info["source_rows"]:
            frame_set.add(int(r["frame_idx"]))
    frame_list = sorted(frame_set)
    print(f"[agg] {len(frame_list)} distinct frames across {len(objects)} object(s)", flush=True)

    lut = {}
    for oid, info in objects.items():
        for r in info["source_rows"]:
            lut[(oid, int(r["frame_idx"]))] = r

    sidecar = {
        "trial_dir": args.trial, "csv": demo_csv, "image_dir": src_img_dir,
        "events": events,
        "objects": {str(oid): {"role": info["role"], "color": list(info["color"])}
                    for oid, info in objects.items()},
        "frames": [],
    }
    summary_rows = []
    overlay_paths = []

    print("[render] building combined overlays + per-frame records", flush=True)
    for n_done, fidx in enumerate(frame_list):
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
            m = mask_from_overlay(ov_path_single, src_path, info["color"])
            if m is None or m.sum() == 0:
                continue
            layer = np.zeros_like(src)
            layer[m] = info["color"]
            comp = cv2.addWeighted(comp, 1.0, layer, 0.5, 0)
            bb = bbox(m)
            per_obj.append({"obj_id": oid, "role": info["role"],
                            "mask_px": int(m.sum()), "bbox_xyxy": bb})
        cv2.putText(comp, f"f{fidx:03d}  {png}  ({len(per_obj)} obj)", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        ov_out = os.path.join(out_dir, "overlays", f"f{fidx:04d}_{png}")
        cv2.imwrite(ov_out, comp)
        overlay_paths.append(ov_out)

        img_id_str = png.replace(".png", "")
        sidecar["frames"].append({
            "frame_idx": fidx, "img_filename": png,
            "t_rel_s": img_to_trel.get(img_id_str),
            "overlay_path": ov_out, "objects": per_obj,
        })
        for o in per_obj:
            bb = o["bbox_xyxy"] or [-1, -1, -1, -1]
            summary_rows.append([fidx, png, o["obj_id"], o["role"], o["mask_px"], *bb, ov_out])
        if (n_done + 1) % 100 == 0 or n_done == 0:
            print(f"  [render] {n_done+1}/{len(frame_list)} frames composed", flush=True)

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

    if overlay_paths:
        sample = cv2.imread(overlay_paths[0])
        h, w = sample.shape[:2]
        mp4 = os.path.join(out_dir, "overlay.mp4")
        backup_if_exists(mp4)
        vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
        print(f"[video] writing {mp4} ({len(overlay_paths)} frames @ 15 fps)", flush=True)
        for i, p in enumerate(overlay_paths):
            vw.write(cv2.imread(p))
            if (i + 1) % 100 == 0:
                print(f"  [video] {i+1}/{len(overlay_paths)}", flush=True)
        vw.release()
        print(f"[video] {mp4}", flush=True)

    print(f"[done] {len(objects)} object(s) composed into {json_path}", flush=True)


if __name__ == "__main__":
    main()
