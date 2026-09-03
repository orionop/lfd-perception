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
import csv
import json
import os
import shutil

import cv2
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deliverable_events import detect_events, load_demo_rows
from event_utils import mask_from_overlay

IMG_DIR_NAME = "zed_zed_node_rgb_color_rect_image_compressed"

POSE_TS = "NS_1.franka_robot_state_broadcaster.current_pose.timestamp"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"


def backup_if_exists(path):
    if os.path.exists(path):
        bak = path + ".bak"
        shutil.copy2(path, bak)
        print(f"[backup] {path} -> {bak}", flush=True)


def events_from_demo(csv_path):
    """Canonical multi-cycle events, keyed compatibly for the sidecar."""
    raw, _, _ = detect_events(load_demo_rows(csv_path))
    totals = {name: sum(e["event"] == name for e in raw)
              for name in ("grasp", "press", "release")}
    seen = {name: 0 for name in totals}
    out = {}
    for event in raw:
        name = event["event"]
        seen[name] += 1
        key = name if totals[name] == 1 else f"{name}_{seen[name]}"
        out[key] = {k: event[k] for k in
                    ("t_rel_s", "row_idx", "img_ts", "force_mag_n",
                     "gripper_width_m")}
    return out


# mask_from_overlay now lives in Code/event_utils.py -- the local copy
# recovered the overlay's caption text as object pixels because the
# propagation scripts draw that caption in the object's own colour.


CAPTION_COLOR = (255, 255, 255)


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
    ap.add_argument("--selection-report", default=None,
                    help="selection_report.json; review_required is rejected")
    ap.add_argument("--portable-paths", action="store_true",
                    help="store JSON paths relative to the sidecar directory")
    args = ap.parse_args()

    # Deterministic: os.listdir order is arbitrary, so a trial directory with
    # more than one CSV silently picked a different file between runs. Prefer
    # the merged "<trial>_0.csv" the schema specifies, then fall back to the
    # first CSV in sorted order.
    cands = sorted(f for f in os.listdir(args.trial)
                   if f.endswith(".csv") and not f.startswith("."))
    prefer = [f for f in cands if f.endswith("_0.csv")]
    demo_csv = (os.path.join(args.trial, (prefer or cands)[0])
                if cands else None)
    if len(cands) > 1:
        print(f"[warn] {len(cands)} CSVs in {args.trial}; using "
              f"{os.path.basename(demo_csv)}", flush=True)
    if demo_csv is None:
        raise FileNotFoundError(f"no merged CSV in {args.trial}")
    src_img_dir = os.path.join(args.trial, IMG_DIR_NAME)
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "overlays"), exist_ok=True)

    selection = None
    if args.selection_report:
        with open(args.selection_report) as f:
            selection = json.load(f)
        if selection.get("status") == "review_required":
            raise RuntimeError("selection requires review; refusing to publish objects.json")

    def json_path(path):
        if not args.portable_paths or path is None:
            return path
        return os.path.relpath(os.path.abspath(path), os.path.abspath(out_dir))

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

    # BUG 5 (found 2026-08-30). A propagation run's overlay is not guaranteed
    # to be drawn in the colour this sidecar declares for that role:
    # figures/propagation_obj4_charger was rendered with
    # propagate_object_n.py's DEFAULT --color (0,165,255) while objects.json
    # declares charger_contact as (0,215,255). Recovery survived only because
    # the per-channel test reduces each channel to one bit, so the two colours
    # share a predicate. That is luck, not correctness, so detect the colour
    # each run actually used, recover with it, and say so.
    #
    # Overlays are composited as img*1.0 + layer*alpha with alpha=0.5, so on
    # mask pixels diff ~= 0.5*colour and 2*median(diff) recovers it. Channels
    # that clip at 255 read low, which is why the check below is a signature
    # comparison rather than an equality test.
    def detect_color(rows_):
        best = max(rows_, key=lambda r: float(r.get("mask_px", 0) or 0))
        if float(best.get("mask_px", 0) or 0) <= 0:
            return None
        ov = cv2.imread(best["overlay_path"])
        sp = os.path.join(src_img_dir, best["file"])
        sr = cv2.imread(sp)
        if ov is None or sr is None or ov.shape != sr.shape:
            return None
        d = ov.astype(int) - sr.astype(int)
        strong = np.abs(d).sum(axis=2) > 60
        if not strong.any():
            return None
        return np.clip(2 * np.median(d[strong], axis=0), 0, 255)

    for oid, info in objects.items():
        det = detect_color(info["source_rows"])
        info["source_color"] = None if det is None else [int(v) for v in det]
        if det is None:
            print(f"  [warn] obj_id={oid}: could not detect overlay colour; "
                  f"using declared {info['color']}", flush=True)
            continue
        sig_d = tuple(c > 100 for c in info["color"])
        sig_a = tuple(c > 100 for c in det)
        same = np.allclose(det, info["color"], atol=40)
        if not same:
            print(f"  [warn] obj_id={oid} role={info['role']}: overlays were "
                  f"drawn in ~{info['source_color']}, but this sidecar "
                  f"declares {list(info['color'])}. Recovery uses the drawn "
                  f"colour; display uses the declared one. Signatures "
                  f"{'agree' if sig_d == sig_a else 'DISAGREE'}.", flush=True)

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
        "schema_version": "1.0",
        "path_base": "sidecar_directory" if args.portable_paths else "working_directory",
        "trial_dir": json_path(args.trial), "csv": json_path(demo_csv),
        "image_dir": json_path(src_img_dir),
        "run": {
            "status": (selection or {}).get("status", "manual"),
            "automatic": bool((selection or {}).get("automatic", False)),
            "selection_report": json_path(args.selection_report),
        },
        "events": events,
        "objects": {str(oid): {"role": info["role"],
                               "color": list(info["color"]),
                               "source_color": info.get("source_color")}
                    for oid, info in objects.items()},
        "frames": [],
    }
    summary_rows = []
    overlay_paths = []

    print("[render] building combined overlays + per-frame records", flush=True)
    for n_done, fidx in enumerate(frame_list):
        present = [lut[(oid, fidx)] for oid in objects if (oid, fidx) in lut]
        if not present:
            continue
        # Every object's propagation run must agree on which PNG this
        # frame_idx refers to. Taking whichever row dict order reached first
        # would compose the overlay on the wrong source image if two runs
        # were produced against different frame sets.
        names = {r["file"] for r in present}
        if len(names) > 1:
            print(f"  [skip] frame {fidx}: objects disagree on the source "
                  f"frame {sorted(names)}", flush=True)
            continue
        any_row = present[0]
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
            # NO other_colors here, deliberately. This overlay is the
            # SINGLE-OBJECT one written by the propagation run, so there is
            # nothing to disambiguate against, and doing so is harmful:
            # figures/propagation_obj4_charger was rendered with
            # propagate_object_n.py's DEFAULT --color (0,165,255) rather than
            # the (0,215,255) this sidecar declares for charger_contact
            # (verified from the overlays: median 2*diff = (0,164,254)).
            # Nearest-colour disambiguation therefore assigned every charger
            # pixel to tool_contact and emptied the role. The declared colour
            # is used for DISPLAY in the combined overlay; the loose
            # per-channel test is what recovers the mask, and with one object
            # per image that is sufficient.
            # recover with the colour the overlay was actually drawn in
            m = mask_from_overlay(ov_path_single, src_path,
                                  info.get("source_color") or info["color"])
            if m is None or m.sum() == 0:
                continue
            layer = np.zeros_like(src)
            layer[m] = info["color"]
            comp = cv2.addWeighted(comp, 1.0, layer, 0.5, 0)
            bb = bbox(m)
            # mask_px comes from the PROPAGATION SUMMARY, not the overlay.
            #
            # Overlay recovery is inherently lossy on bright objects: masks
            # are alpha-blended at 0.5, so a pixel whose source channel is
            # already near 255 saturates and its diff falls below tolerance.
            # The recovered count is therefore a documented LOWER BOUND (see
            # Code/event_utils.py). The propagation summary row already in
            # `lut` carries mask.sum() taken directly from the SAM 2 output,
            # which is authoritative and costs nothing to use.
            #
            # The recovered value is kept alongside as mask_px_overlay so the
            # discrepancy stays visible rather than being silently dropped.
            src_row = lut[(oid, fidx)]
            recovered = int(m.sum())
            try:
                authoritative = int(float(src_row["mask_px"]))
            except (KeyError, ValueError):
                authoritative = recovered
            per_obj.append({"obj_id": oid, "role": info["role"],
                            "mask_px": authoritative,
                            "mask_px_overlay": recovered,
                            "bbox_xyxy": bb,
                            "object_overlay_path": json_path(ov_path_single)})
        # White, never an object colour. Yellow (0,255,255) was previously used
        # here and sits one channel away from the charger_contact gold
        # (0,215,255), i.e. the same class of bug already fixed in the
        # propagation scripts, where a caption drawn in an object's colour was
        # recovered as object pixels.
        cv2.putText(comp, f"f{fidx:03d}  {png}  ({len(per_obj)} obj)", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, CAPTION_COLOR, 2)
        ov_out = os.path.join(out_dir, "overlays", f"f{fidx:04d}_{png}")
        cv2.imwrite(ov_out, comp)
        overlay_paths.append(ov_out)

        img_id_str = png.replace(".png", "")
        sidecar["frames"].append({
            "frame_idx": fidx, "img_filename": png,
            "t_rel_s": img_to_trel.get(img_id_str),
            "overlay_path": json_path(ov_out), "objects": per_obj,
        })
        for o in per_obj:
            bb = o["bbox_xyxy"] or [-1, -1, -1, -1]
            # overlay_path is the SINGLE-OBJECT overlay, because every
            # consumer of this column uses it to recover THIS row's mask.
            #
            # BUG 4 (fixed 2026-08-30): it used to be the combined overlay,
            # so a row describing one object pointed at an image containing
            # all of them. Combined with the colour-signature collision in
            # event_utils.mask_from_overlay, tool_contact and charger_contact
            # recovered byte-identical masks. The composited image is still
            # recorded, as combined_overlay_path, for display and video.
            summary_rows.append([fidx, png, o["obj_id"], o["role"],
                                 o["mask_px"], o["mask_px_overlay"], *bb,
                                 lut[(o["obj_id"], fidx)]["overlay_path"], ov_out])
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
                    "mask_px_overlay",
                    "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "overlay_path",
                    "combined_overlay_path"])
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
