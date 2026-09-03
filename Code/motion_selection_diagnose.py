"""
Why did two of the five trials fail the image-stationarity test?

Code/motion_selection_probe.py measured, for the grasped object, how far its
image centroid travels per metre of end effector travel, inside versus outside
the gripper-closed window. Three trials separated cleanly:

    lfdws_t005            ratio 0.036   static frames while held 98.0%
    lfdws_t002_labexport  ratio 0.049                            93.8%
    lfdws_t002_new        ratio 0.080                            94.1%
    lfdws_t004            ratio 0.796                            57.2%   <-- ?
    lfdws_t001            ratio 1.248                            64.6%   <-- ?

Two hypotheses were stated but never tested. This script tests them rather
than assuming them, because "the two failures have innocent explanations" is
exactly the kind of claim that is comfortable and wrong.

H1, lfdws_t004: THE GRIPPER IS CLOSED BUT EMPTY.
    The arm travels 1.277 m during the closed window, which a gripper rigidly
    holding a wall-mounted pegboard latch could not permit. The decisive test
    is the gripper width itself: closing on an object parks the width at that
    object's thickness, closing on nothing drives it to roughly zero. If the
    closed window is mostly near-zero width, the window is not a hold, and the
    ratio is not measuring what the probe assumed.

    This would be a finding rather than only a failure. A near-1 ratio while
    the gripper reports closed is the signature of an empty grasp, which is a
    detector the pipeline does not currently have.

H2, lfdws_t001: THIN SAMPLE, AND POSSIBLY THE WRONG SOURCE TRACK.
    Only 62 not-held steps over 0.383 m, against 175 held steps. Separately,
    figures/ holds more than one propagation for this trial and the filename
    does not say which is current (the documented multiple-propagations trap:
    figures/propagation_summary.csv is forward-only, propagation_bidir_summary
    .csv is the bidirectional rerun). If the sidecar was built from the
    superseded track, the centroid series is not the object's.

Also computes, for every trial, a per-segment breakdown of the closed window.
A trial that genuinely holds for part of the window and travels empty for the
rest will show both signatures, which a single whole-window ratio hides.

Read only. Touches no pipeline output.

Usage:
    .venv_analysis/bin/python Code/motion_selection_diagnose.py
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from event_utils import (MIN_GRIPPER_SPAN_M, gripper_closed_window,
                         parse_gripper_width)

POSE_X = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.x"
POSE_Y = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.y"
POSE_Z = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.z"
GRIP = "NS_1.franka_gripper.joint_states.position"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"

OUT_CSV = "figures/motion_selection_diagnose.csv"

# A Franka Hand closing on nothing parks near zero. Anything below this is
# treated as an empty grasp rather than a held object.
EMPTY_GRASP_WIDTH_M = 0.004

TRIALS = [
    ("lfdws_t001", "Data/lfdws_t001/lfdws_t001/lfdws_t001_0.csv",
     "figures/identify/objects_summary.csv", "grasped"),
    ("lfdws_t002_new", "Data/lfdws_t002_new/lfdws_t002_new_0.csv",
     "figures/t002new/identify/objects_summary.csv", "grasped"),
    ("lfdws_t002_labexport",
     "Data/lfdws_t002_labexport/lfdws_t002/lfdws_t002.csv",
     "figures/t002labexport/identify/objects_summary.csv", "grasped"),
    ("lfdws_t004", "Data/lfdws_t004/lfdws_t004_0.csv",
     "figures/t004/identify/objects_summary.csv", "grasped"),
    ("lfdws_t005", "Data/lfdws_t005/lfdws_t005_0.csv",
     "figures/t005/identify/objects_summary.csv", "grasped"),
]

# Candidate propagation tracks per trial, for the H2 source-track check.
CANDIDATE_TRACKS = {
    "lfdws_t001": ["figures/propagation_summary.csv",
                   "figures/propagation_bidir_summary.csv"],
    "lfdws_t002_new": ["figures/t002new/propagation_grasped_summary.csv",
                       "figures/t002new/propagation_grasped_box_summary.csv"],
    "lfdws_t002_labexport": [
        "figures/t002labexport/propagation_grasped_summary.csv",
        "figures/t002labexport/propagation_grasped_pointprompt_summary.csv"],
}


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_track(path, role):
    out = {}
    for row in load_rows(path):
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
        out[os.path.splitext(row["img_filename"])[0]] = (
            0.5 * (x0 + x1), 0.5 * (y0 + y1), px)
    return out


def collapse_to_frames(rows, closed, ee, img_ids):
    frames, seen = [], set()
    for i, iid in enumerate(img_ids):
        if iid in seen:
            continue
        seen.add(iid)
        frames.append((iid, ee[i], bool(closed[i]), i))
    return frames


def segment_ratio(frames, track, lo, hi):
    """Aggregate pixels-per-metre over frames whose row index is in [lo,hi)."""
    dpx, dee = 0.0, 0.0
    n = 0
    for k in range(len(frames) - 1):
        _, ee_a, _, ia = frames[k]
        _, ee_b, _, ib = frames[k + 1]
        if not (lo <= ia < hi and lo <= ib < hi):
            continue
        a = track.get(frames[k][0])
        b = track.get(frames[k + 1][0])
        if a is None or b is None:
            continue
        d = float(np.linalg.norm(ee_b - ee_a))
        if d <= 1e-4:
            continue
        dpx += float(np.hypot(b[0] - a[0], b[1] - a[1]))
        dee += d
        n += 1
    return (dpx / dee if dee > 0 else float("nan")), n, dee


def h1_empty_grasp(label, w, closed):
    """Is the 'closed' window actually a hold, or a gripper shut on nothing?"""
    wc = w[closed]
    if len(wc) == 0:
        return None
    frac_empty = float((wc < EMPTY_GRASP_WIDTH_M).mean())
    print(f"  [H1] closed-window width: median {np.nanmedian(wc)*1000:7.3f} mm"
          f"   min {np.nanmin(wc)*1000:7.3f}   max {np.nanmax(wc)*1000:7.3f}",
          flush=True)
    print(f"  [H1] fraction of closed samples below "
          f"{EMPTY_GRASP_WIDTH_M*1000:.0f} mm (empty grasp): "
          f"{frac_empty:.1%}", flush=True)
    if frac_empty > 0.5:
        print("  [H1] VERDICT: the closed window is mostly an EMPTY grasp. "
              "The stationarity ratio is not measuring a held object here.",
              flush=True)
    else:
        print("  [H1] VERDICT: the gripper is holding something through most "
              "of the window, so an empty grasp does NOT explain this trial.",
              flush=True)
    return frac_empty


def h2_source_track(label, track):
    """Does the sidecar's mask area match one propagation track or another?"""
    cands = CANDIDATE_TRACKS.get(label)
    if not cands:
        return
    side_px = np.array([v[2] for v in track.values()], dtype=float)
    if len(side_px) == 0:
        return
    print(f"  [H2] sidecar mask_px: mean {side_px.mean():9.1f}  "
          f"n={len(side_px)}", flush=True)
    for c in cands:
        if not os.path.exists(c):
            print(f"  [H2]   {os.path.basename(c):48s} MISSING", flush=True)
            continue
        px = np.array([float(r["mask_px"]) for r in load_rows(c)
                       if float(r["mask_px"]) > 0], dtype=float)
        if len(px) == 0:
            continue
        ratio = side_px.mean() / px.mean() if px.mean() else float("nan")
        flag = "  <-- matches" if 0.7 < ratio < 1.4 else ""
        print(f"  [H2]   {os.path.basename(c):48s} mean {px.mean():9.1f}  "
              f"ratio {ratio:5.2f}{flag}", flush=True)


def main():
    print("[diagnose] why two trials failed the stationarity test", flush=True)
    print(f"[diagnose] empty-grasp width threshold "
          f"{EMPTY_GRASP_WIDTH_M*1000:.0f} mm; gripper span guard "
          f"{MIN_GRIPPER_SPAN_M*1000:.0f} mm\n", flush=True)

    out_rows = []
    for label, csv_path, sidecar, role in TRIALS:
        print(f"[trial] {label}", flush=True)
        if not (os.path.exists(csv_path) and os.path.exists(sidecar)):
            print("  [skip] missing csv or sidecar\n", flush=True)
            continue
        rows = load_rows(csv_path)
        if GRIP not in rows[0]:
            print("  [skip] no gripper topic\n", flush=True)
            continue
        w = np.array([parse_gripper_width(r[GRIP]) for r in rows])
        closed = gripper_closed_window(w)
        if not closed.any():
            print("  [skip] span guard tripped, gripper never actuated\n",
                  flush=True)
            continue
        ee = np.array([[float(r[POSE_X]), float(r[POSE_Y]), float(r[POSE_Z])]
                       for r in rows])
        img_ids = [str(r[IMG]) for r in rows]
        track = load_track(sidecar, role)
        frames = collapse_to_frames(rows, closed, ee, img_ids)
        print(f"  [load] {len(rows)} pose rows -> {len(frames)} image frames, "
              f"{len(track)} tracked", flush=True)

        frac_empty = h1_empty_grasp(label, w, closed)
        h2_source_track(label, track)

        # per-segment breakdown of the closed window: a trial that holds for
        # part of it and travels empty for the rest hides that in a single
        # whole-window ratio
        idx = np.flatnonzero(closed)
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        nseg = 4
        edges = np.linspace(lo, hi, nseg + 1).astype(int)
        print(f"  [seg] closed window rows {lo}..{hi}, split into {nseg}:",
              flush=True)
        for s in range(nseg):
            r, n, travel = segment_ratio(frames, track, edges[s], edges[s + 1])
            wseg = w[edges[s]:edges[s + 1]]
            wmed = float(np.nanmedian(wseg)) * 1000 if len(wseg) else float("nan")
            print(f"    seg{s+1}  px/m {r:9.1f}   steps {n:4d}   "
                  f"ee travel {travel:6.3f} m   median width {wmed:7.3f} mm",
                  flush=True)
            out_rows.append([label, f"seg{s+1}", f"{r:.3f}", n,
                             f"{travel:.4f}", f"{wmed:.4f}",
                             f"{frac_empty:.4f}" if frac_empty is not None
                             else ""])
        print("", flush=True)

    os.makedirs("figures", exist_ok=True)
    if os.path.exists(OUT_CSV):
        import shutil
        shutil.copy2(OUT_CSV, OUT_CSV + ".bak")
    with open(OUT_CSV, "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["trial", "segment", "px_per_m", "steps", "ee_travel_m",
                      "median_width_mm", "closed_frac_empty"])
        wtr.writerows(out_rows)
    print(f"[write] {OUT_CSV}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
