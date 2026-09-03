"""
MEASURE where a held object sits, instead of deriving it from CAD.

WHY MEASURE
-----------
Geometric seeding of the contact roles works (Code/geometric_seed.py, 5/7,
against the heuristic's 0/6) and is confirmed end to end (0.945 track
reproduction, Code/seed_e2e_check.py). The grasped role is the one gap: it
needs the gripper centre in the bota frame, and two derivations have failed.

  * six signed axis directions: none lands inside the grasped mask, best
    250.7 px away
  * the 1-DOF family constrained to map Mark's global camera vector onto the
    verified bota camera vector: 0 of 1440 angles work, 0/5 held out

Both failed for the same reason: the rotation from Mark's drawing frame to
the bota frame is the step his scan omits, and no convention reproduces it.

But the quantity does not have to be derived. The camera is bolted to the
same body as the gripper, so an object that is genuinely held sits at a FIXED
position in the camera frame. That position is directly observable: take the
grasped mask on a depth-bearing trial, back-project it through real depth,
and push it into the bota frame with the already-validated T_bota_camera.

No CAD chain, no frame relation, nothing from the lab. And a measurement of
where held objects actually are is better evidence for seeding them than a
number on paper.

THE BUG THIS FIXES
------------------
A first attempt returned only one usable sample. The cause was the same
defect already fixed in Code/contact_eval_set.py: the depth stream runs on
its own timeline, so the row-matched depth frame id is NOT the rgb frame id
except in the lab's native export. Passing the rgb id made every depth lookup
silently miss. This builds the rgb -> depth id map from the merged CSV.

VALIDATION
----------
Leave-one-trial-out. Measure the offset on all but one recording, then check
whether it projects inside the grasped mask on the recording it never saw. A
measurement that only works where it was taken is not a measurement.

Read only. Writes one small JSON that Code/geometric_seed.py picks up.

Usage:
    .venv_analysis/bin/python Code/measure_grasp_offset.py
"""
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from event_utils import (gripper_closed_window, mask_from_overlay,
                         parse_gripper_width)
from geometric_seed import load_calib, project_bota_point

GRIP = "NS_1.franka_gripper.joint_states.position"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"
DEPTH_COL = "zed.zed_node.depth.depth_registered.compressedDepth"
POSE = ["NS_1.franka_robot_state_broadcaster.current_pose.pose.position.x",
        "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.y",
        "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.z",
        "NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation.x",
        "NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation.y",
        "NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation.z",
        "NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation.w"]
RGB = "zed_zed_node_rgb_color_rect_image_compressed"
DEP = "zed_zed_node_depth_depth_registered_compressedDepth"
GRASPED_BGR = (0, 255, 0)
OUT_JSON = "figures/grasp_offset.json"

# Samples taken from inside each hold, avoiding the first and last 20 percent
# so the fingers have closed and not yet opened.
N_SAMPLES = 6

TRIALS = [
    ("lfdws_t002_new", "Data/lfdws_t002_new",
     "figures/t002new/identify/objects_summary.csv"),
    ("lfdws_t002_labexport", "Data/lfdws_t002_labexport/lfdws_t002",
     "figures/t002labexport/identify/objects_summary.csv"),
    ("lfdws_t004", "Data/lfdws_t004",
     "figures/t004/identify/objects_summary.csv"),
    ("lfdws_t005", "Data/lfdws_t005",
     "figures/t005/identify/objects_summary.csv"),
]


def find_csv(d):
    for f in sorted(os.listdir(d)):
        if f.endswith(".csv") and not f.startswith("."):
            return os.path.join(d, f)
    return None


def load_depth(tdir, depth_id, H, W):
    # ORDER MATTERS. The lab's native export writes BOTH a float32-metre .npy
    # and an 8-bit uint8 .png preview of the same frame; only the .npy is
    # depth data. Reading the .png first and dividing by 1000 yields a median
    # of exactly 0.255 m on every frame (255/1000, saturated), which is what
    # corrupted the lfdws_t002_labexport samples. Our own mcap_extract.py
    # writes a 16-bit millimetre .png and no .npy, so .png stays the fallback.
    for ext in (".npy", ".png"):
        p = os.path.join(tdir, DEP, f"{depth_id}{ext}")
        if os.path.exists(p):
            if ext == ".npy":
                d = np.load(p).astype(np.float32)
            else:
                d = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                if d is None:
                    return None
                d = d.astype(np.float32) / 1000.0
            if d.shape != (H, W):
                d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
            return d
    return None


def collect(label, tdir, sidecar, K, T_bc):
    cpath = find_csv(tdir)
    if not (cpath and os.path.exists(sidecar)):
        print(f"[skip] {label}: missing inputs", flush=True)
        return []
    rows = list(csv.DictReader(open(cpath)))
    if GRIP not in rows[0]:
        print(f"[skip] {label}: no gripper topic", flush=True)
        return []
    w = np.array([parse_gripper_width(r[GRIP]) for r in rows])
    closed = gripper_closed_window(w)
    if not closed.any():
        print(f"[skip] {label}: span guard tripped", flush=True)
        return []

    # one entry per distinct image frame, carrying the ROW-MATCHED depth id
    frames, seen = [], set()
    for i, r in enumerate(rows):
        iid = str(r[IMG])
        if iid in seen:
            continue
        seen.add(iid)
        frames.append({"img": iid,
                       "dep": str(r[DEPTH_COL]) if DEPTH_COL in r else iid,
                       "closed": bool(closed[i]),
                       "pose": np.array([float(r[k]) for k in POSE])})
    idx = [i for i, f in enumerate(frames) if f["closed"]]
    if len(idx) < 10:
        print(f"[skip] {label}: hold too short", flush=True)
        return []
    lo, hi = idx[int(0.2 * len(idx))], idx[int(0.8 * len(idx))]
    picks = np.linspace(lo, hi, N_SAMPLES).astype(int)

    track = {os.path.splitext(r["img_filename"])[0]: r
             for r in csv.DictReader(open(sidecar))
             if r["role"] == "grasped" and float(r["mask_px"]) > 0}

    out = []
    for p in picks:
        f = frames[p]
        if f["img"] not in track:
            continue
        rgb = os.path.join(tdir, RGB, f"{f['img']}.png")
        if not os.path.exists(rgb):
            continue
        m = mask_from_overlay(track[f["img"]]["overlay_path"], rgb,
                              GRASPED_BGR)
        if m is None or not m.any():
            continue
        H, W = m.shape
        d = load_depth(tdir, f["dep"], H, W)
        if d is None:
            continue
        # Reject samples that are not a held object, on physical grounds set
        # BEFORE looking at the outcome:
        #   depth band -- this workspace is compressed to roughly 0.15-0.21 m
        #     (the finding behind the depth-ranking failure). A "held object"
        #     at 0.45 m is not in the gripper.
        #   mask size -- a held object fills a decent part of the near view.
        #     lfdws_t004 contributed 592-1167 px masks at 0.31-0.45 m, which
        #     are the drifted track and its empty-grasp cycle, not a hold.
        sel = m & (d > 0.08) & (d < 0.26)
        if sel.sum() < 2000:
            continue
        ys, xs = np.nonzero(sel)
        Z = d[sel]
        X = (xs - K[0, 2]) / K[0, 0] * Z
        Y = (ys - K[1, 2]) / K[1, 1] * Z
        # median 3D point of the mask, robust to depth speckle
        p_cam = np.array([np.median(X), np.median(Y), np.median(Z), 1.0])
        p_bota = (T_bc @ p_cam)[:3]
        out.append({"trial": label, "img": f["img"], "pose": f["pose"],
                    "mask": m, "p_bota": p_bota,
                    "n_px": int(sel.sum()), "Z": float(np.median(Z))})
    print(f"[collect] {label}: {len(out)} usable samples", flush=True)
    return out


def main():
    K, T_bc = load_calib()
    print("[measure] where does a held object sit in the bota frame?\n",
          flush=True)
    samples = []
    for label, tdir, side in TRIALS:
        samples += collect(label, tdir, side, K, T_bc)
    if len(samples) < 4:
        print(f"\n[fatal] only {len(samples)} samples, not enough", flush=True)
        return

    print(f"\n{'trial':24s} {'depth m':>8s} {'px':>7s} "
          f"{'position in bota frame (mm)':>32s}", flush=True)
    for s in samples:
        print(f"{s['trial']:24s} {s['Z']:8.3f} {s['n_px']:7d} "
              f"{np.round(s['p_bota']*1000,1).tolist()!s:>32s}", flush=True)

    P = np.array([s["p_bota"] for s in samples])
    print(f"\n[all] mean   {np.round(P.mean(0)*1000,1).tolist()} mm"
          f"   |mean| {np.linalg.norm(P.mean(0))*1000:.1f} mm", flush=True)
    print(f"[all] spread {np.round(P.std(0)*1000,1).tolist()} mm", flush=True)

    trials = sorted(set(s["trial"] for s in samples))
    print(f"\n[per trial]", flush=True)
    for t in trials:
        Q = np.array([s["p_bota"] for s in samples if s["trial"] == t])
        print(f"  {t:24s} n={len(Q)}  mean "
              f"{np.round(Q.mean(0)*1000,1).tolist()}  "
              f"spread {np.round(Q.std(0)*1000,1).tolist()}", flush=True)

    print(f"\n[loto] leave-one-trial-out: measure on the rest, test on the "
          f"held-out recording", flush=True)
    held_ok = held_n = 0
    for t in trials:
        train = np.array([s["p_bota"] for s in samples if s["trial"] != t])
        off = train.mean(0)
        ok = n = 0
        for s in samples:
            if s["trial"] != t:
                continue
            uv, z = project_bota_point(off, s["pose"], K, T_bc)
            H, W = s["mask"].shape
            n += 1
            if z > 0 and 0 <= uv[0] < W and 0 <= uv[1] < H and \
                    s["mask"][int(uv[1]), int(uv[0])]:
                ok += 1
        held_ok += ok
        held_n += n
        print(f"  held out {t:24s} {ok}/{n} samples land inside the grasped "
              f"mask", flush=True)
    print(f"\n[loto] total {held_ok}/{held_n}", flush=True)

    off = P.mean(0)
    if held_n and held_ok / held_n >= 0.6:
        os.makedirs("figures", exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump({"offset_bota_m": off.tolist(),
                       "n_samples": len(samples), "trials": trials,
                       "loto_inside": held_ok, "loto_total": held_n,
                       "spread_mm": (P.std(0) * 1000).tolist()}, f, indent=2)
        print(f"[write] {OUT_JSON}", flush=True)
        print("[verdict] MEASURED and generalises across recordings. The "
              "grasped role can be seeded without any further input from the "
              "lab.", flush=True)
    else:
        print("[verdict] does NOT generalise. Held objects do not sit at a "
              "consistent place in the bota frame across recordings, so a "
              "single fixed offset cannot seed them.", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
