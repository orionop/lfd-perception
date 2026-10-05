"""
Is "stationary in the image while the arm moves" a calibration-free selection
signal for the grasped object?

Hypothesis. The ZED is mounted eye-in-hand, rigidly attached to the same body
as the gripper. So while an object is actually held, it is rigidly attached to
the camera too, and its projection must be near-stationary in the image no
matter how far the arm travels. Every static world object, by contrast, sweeps
across the image driven by camera ego-motion. If that holds, then during the
gripper-closed window the task-relevant grasped object is identifiable as the
one that does NOT move in the image, which requires no camera extrinsic at all
and therefore is not blocked on T_bota_camera.

This matters because the measured failure of label-free selection
(baseline_sam_depth_ranking.py: oracle IoU 0.787 median, achieved 0.104, and
41 percent of proposals sitting within 2 cm of the target in depth) says the
missing ingredient is a selection signal, not a better segmenter. Image
stationarity under known ego-motion is a candidate for that signal.

What this script measures, per trial, using only artifacts that already exist:

  1. end effector path length inside vs outside the gripper-closed window,
     from current_pose. This is the "the arm really did move" control. If the
     arm barely moves while closed, the trial cannot test anything.
  2. the tracked grasped object's bbox centroid displacement per frame, inside
     vs outside that window, from the sidecar objects_summary.csv.
  3. the ratio of the two. The hypothesis predicts centroid motion per unit of
     end effector travel collapses once the object is held.

Deliberately NOT measured here: background optical flow. That needs a pass
over the raw frames and belongs in a follow-up if this cheap check survives.
The control in (1) is a weaker but free stand-in, since large end effector
travel with a static image projection already implies the background moved.

Read only. Touches no pipeline output.

Usage:
    .venv_analysis/bin/python Code/motion_selection_probe.py
"""
import ast
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from event_utils import gripper_closed_window, parse_gripper_width

POSE_X = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.x"
POSE_Y = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.y"
POSE_Z = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.z"
GRIP = "NS_1.franka_gripper.joint_states.position"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"

# Minimum end effector travel for one frame-to-frame step to count. See the
# comment in probe() for the measurements behind this value.
MIN_STEP_TRAVEL_M = 0.002

# (label, merged csv, sidecar objects_summary.csv, role to probe)
TRIALS = [
    ("lfdws_t001",
     "Data/lfdws_t001/lfdws_t001/lfdws_t001_0.csv",
     "figures/identify/objects_summary.csv", "grasped"),
    ("lfdws_t002_new",
     "Data/lfdws_t002_new/lfdws_t002_new_0.csv",
     "figures/t002new/identify/objects_summary.csv", "grasped"),
    ("lfdws_t002_labexport",
     "Data/lfdws_t002_labexport/lfdws_t002/lfdws_t002.csv",
     "figures/t002labexport/identify/objects_summary.csv", "grasped"),
    ("lfdws_t004",
     "Data/lfdws_t004/lfdws_t004_0.csv",
     "figures/t004/identify/objects_summary.csv", "grasped"),
    ("lfdws_t005",
     "Data/lfdws_t005/lfdws_t005_0.csv",
     "figures/t005/identify/objects_summary.csv", "grasped"),
]


def find_csv(trial_dir):
    for f in sorted(os.listdir(trial_dir)):
        if f.endswith(".csv"):
            return os.path.join(trial_dir, f)
    return None


def load_merged(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_sidecar(path, role):
    """{img_id_without_ext: (cx, cy, mask_px)} for the requested role."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["role"] != role:
                continue
            try:
                x0, y0 = float(row["bbox_x0"]), float(row["bbox_y0"])
                x1, y1 = float(row["bbox_x1"]), float(row["bbox_y1"])
                px = int(float(row["mask_px"]))
            except (ValueError, KeyError):
                continue
            if px <= 0 or x0 < 0:
                continue
            key = os.path.splitext(row["img_filename"])[0]
            out[key] = (0.5 * (x0 + x1), 0.5 * (y0 + y1), px)
    return out


def probe(label, merged_csv, sidecar_csv, role):
    print(f"\n[trial] {label}", flush=True)
    if merged_csv is None or not os.path.exists(str(merged_csv)):
        print("  [skip] merged CSV not found", flush=True)
        return None
    if not os.path.exists(sidecar_csv):
        print(f"  [skip] sidecar not found: {sidecar_csv}", flush=True)
        return None

    rows = load_merged(merged_csv)
    if GRIP not in rows[0]:
        print("  [skip] no gripper topic, no closed window to test", flush=True)
        return None

    w = np.array([parse_gripper_width(r[GRIP]) for r in rows])
    closed = gripper_closed_window(w)
    if not closed.any():
        print("  [skip] gripper never actuated (span guard tripped)", flush=True)
        return None

    ee = np.array([[float(r[POSE_X]), float(r[POSE_Y]), float(r[POSE_Z])]
                   for r in rows])
    img_ids = [str(r[IMG]) for r in rows]

    track = load_sidecar(sidecar_csv, role)
    print(f"  [load] {len(rows)} pose rows, {int(closed.sum())} closed, "
          f"{len(track)} tracked frames with role={role}", flush=True)

    # The merged CSV is on the pose timeline, which runs far faster than the
    # camera: t004 has 116710 pose rows against 1262 tracked frames, so
    # consecutive pose rows almost always carry the SAME image id and any
    # per-pose-row centroid step is identically zero. Collapse to one entry
    # per distinct image frame first, keeping that frame's first pose row.
    frames = []          # (img_id, ee_xyz, closed)
    seen_at = {}
    for i, iid in enumerate(img_ids):
        if iid in seen_at:
            continue
        seen_at[iid] = i
        frames.append((iid, ee[i], bool(closed[i])))
    print(f"  [collapse] {len(rows)} pose rows -> {len(frames)} distinct "
          f"image frames", flush=True)

    # walk consecutive IMAGE frames, keep only steps where both endpoints have
    # a tracked centroid and the held-state agrees at both ends
    per_state = {True: [], False: []}
    for i in range(len(frames) - 1):
        (id_a, ee_a, cl_a) = frames[i]
        (id_b, ee_b, cl_b) = frames[i + 1]
        if cl_a != cl_b:
            continue
        a, b = track.get(id_a), track.get(id_b)
        if a is None or b is None:
            continue
        d_ee = float(np.linalg.norm(ee_b - ee_a))
        d_px = float(np.hypot(b[0] - a[0], b[1] - a[1]))
        per_state[cl_a].append((d_ee, d_px))

    res = {"trial": label}
    for state, name in [(True, "held"), (False, "not_held")]:
        arr = np.array(per_state[state])
        if len(arr) < 20:
            print(f"  [warn] only {len(arr)} usable steps while {name}", flush=True)
            res[name] = None
            continue
        d_ee, d_px = arr[:, 0], arr[:, 1]
        # Only steps where the arm moved MEANINGFULLY.
        #
        # The original 1e-4 m floor was too permissive and corrupted the
        # aggregate. Code/motion_selection_diagnose.py showed px/m exploding
        # on exactly the low-travel segments: lfdws_t002_new gives 52.8 px/m
        # over a 0.637 m segment but 2559.7 px/m over a 0.006 m one, and
        # lfdws_t005 gives 88.3 over 0.359 m against 2627.2 over 0.230 m of
        # mostly-stationary samples. Those are not the object moving, they
        # are tracker jitter divided by a near-zero denominator. Summing
        # numerator and denominator separately does not save the aggregate,
        # because the jitter still accumulates in the numerator while
        # contributing almost nothing to the denominator.
        #
        # Real motion here runs about 9.5 mm per frame (lfdws_t002_new:
        # 0.637 m over 67 steps), so a 2 mm floor keeps genuine motion and
        # discards the degenerate steps.
        mv = d_ee > MIN_STEP_TRAVEL_M
        ratio = d_px[mv] / d_ee[mv] if mv.any() else np.array([np.nan])
        # median-of-ratios blows up whenever d_ee is near zero, so the
        # aggregate (total pixels swept per total metre travelled) is the
        # statistic to trust; the median is kept only for comparison.
        agg = float(d_px[mv].sum() / d_ee[mv].sum()) if mv.any() else float("nan")
        frac_static = float((d_px[mv] < 2.0).mean()) if mv.any() else float("nan")
        res[name] = {
            "n": int(len(arr)),
            "n_moving": int(mv.sum()),
            "ee_total_m": float(d_ee.sum()),
            "px_per_step_med": float(np.median(d_px)),
            "px_per_m_med": float(np.median(ratio)),
            "px_per_m_agg": agg,
            "frac_static": frac_static,
        }
        print(f"  [{name:8s}] steps={len(arr):5d} moving={int(mv.sum()):5d}  "
              f"ee_travel={d_ee.sum():.3f} m  "
              f"med step={np.median(d_px):6.2f} px  "
              f"AGG px/m={agg:8.1f}  "
              f"frac(<2px)={frac_static:5.1%}", flush=True)

    if res.get("held") and res.get("not_held"):
        a = res["held"]["px_per_m_agg"]
        b = res["not_held"]["px_per_m_agg"]
        if b > 0:
            r = a / b
            print(f"  [ratio] AGG held / not_held = {r:.3f}   "
                  f"({'SUPPORTS' if r < 0.5 else 'does NOT support'} "
                  f"the stationarity hypothesis)", flush=True)
            print(f"          static-frame share: held "
                  f"{res['held']['frac_static']:.1%} vs not_held "
                  f"{res['not_held']['frac_static']:.1%}", flush=True)
            res["ratio"] = r
    return res


def main():
    print("[probe] image stationarity of the held object under end effector "
          "motion", flush=True)
    print("[probe] read only, no pipeline outputs touched", flush=True)
    results = []
    for label, merged, sidecar, role in TRIALS:
        try:
            r = probe(label, merged, sidecar, role)
        except Exception as e:
            print(f"  [error] {label}: {type(e).__name__}: {e}", flush=True)
            r = None
        if r:
            results.append(r)

    print("\n[summary]", flush=True)
    ratios = [r["ratio"] for r in results if r.get("ratio") is not None]
    for r in results:
        if r.get("ratio") is not None:
            print(f"  {r['trial']:24s} held/not_held = {r['ratio']:.3f}",
                  flush=True)
    if ratios:
        print(f"\n  n={len(ratios)} trials, median ratio = "
              f"{np.median(ratios):.3f}", flush=True)
        print("  a ratio well below 1 means the held object barely moves in "
              "the image\n  while the arm travels, which is the "
              "calibration-free selection signal.", flush=True)
    else:
        print("  no trial produced a usable ratio", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
