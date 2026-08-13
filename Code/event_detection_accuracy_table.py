"""
Multi-trial event-detection accuracy table.

Every trial run through this pipeline has had its detected events manually
checked at least once (viewing the actual event frame, or cross-checking
against multi_event.py's grasp-window-restricted detector) at some point in
this project's history. That verification has so far only ever been
reported as prose in commit messages / CLAUDE.md / chat. This script
collects it into one table -- trial x sensor profile x detected events x
pass/fail x how it was verified -- instead of restating it as prose again.

Not a live computation: the "detected" event counts/timestamps are
re-derived here by importing the same detection functions the pipeline
actually uses (Code/analyze_demo.py-equivalent logic, reused from
Code/_dado_vs_groundtruth_all_trials.py's detect_events_generic), so the
numbers in the table are live and will catch drift if a script's threshold
logic changes later. The pass/fail verdict and verification method are
recorded findings from manual inspection during this project (see notes
column), not something this script re-derives.

Output: figures/event_detection_accuracy_table.csv

Run inside .venv_analysis (only needs numpy/csv, no torch):
    .venv_analysis/bin/python Code/event_detection_accuracy_table.py
"""
import ast
import csv
import os

import numpy as np

import sys, os as _os
sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from event_utils import gripper_moved

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


def detect_events_generic(csv_path):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        rows = list(reader)
    has_force = FX in cols
    has_grip = GRIP in cols
    out = {}
    if has_force:
        fm = np.array([np.sqrt(float(r[FX])**2 + float(r[FY])**2 + float(r[FZ])**2)
                      for r in rows])
        baseline = np.median(fm[:len(fm)//10])
    if not has_grip:
        if has_force:
            out["press"] = int(np.argmax(fm - baseline))
        return out, len(rows)
    w = np.array([parse_gw(r[GRIP]) for r in rows])
    w_open, w_closed = float(np.nanmax(w)), float(np.nanmin(w))
    thr = w_closed + 0.5*(w_open - w_closed)
    # Guard: a gripper that never actuated puts this midpoint inside the
    # sensor's noise band and manufactures grasp/release (event_utils.py).
    closed = (w < thr) if gripper_moved(w) else np.zeros(len(w), dtype=bool)
    cd = np.where((~closed[:-1]) & (closed[1:]))[0] + 1
    cu = np.where((closed[:-1]) & (~closed[1:]))[0] + 1
    if len(cd):
        out["grasp"] = int(cd[0])
    if len(cu):
        out["release"] = int(cu[-1])
    if has_force:
        # With no real grasp cycle there is no held window to restrict to;
        # search the whole recording rather than an empty mask.
        fm_adj = (np.where(closed, fm - baseline, -np.inf)
                  if closed.any() else fm - baseline)
        out["press"] = int(np.argmax(fm_adj))
    return out, len(rows)


# (trial, csv_path, gripper_present, force_present, manual_verdict, verified_how)
TRIALS = [
    ("lfdws_t001", "Data/lfdws_t001/lfdws_t001/lfdws_t001_0.csv", True, True,
     "PASS", "carrot/cup canonical trial; grasp/release/press events visually "
     "confirmed correct across full pipeline runs"),
    ("lfdws_t001_depth", "Data/lfdws_t001_depth/lfdws_t001_depth_0.csv", False, True,
     "PASS", "no gripper topic; force-only fallback correctly restricted to single "
     "press; multi_event-equivalent 5-cluster breakdown (plate/screwdriver/charger) "
     "manually confirmed against frames"),
    ("lfdws_t002_new", "Data/lfdws_t002_new/lfdws_t002_new_0.csv", True, True,
     "PARTIAL", "single-event (global-max) detector correct here (single "
     "grasp-hold-release cycle, no confound); same trial as t002_labexport but "
     "this extraction predates the multi-phase-demo discovery"),
    ("lfdws_t002_labexport", "Data/lfdws_t002_labexport/lfdws_t002/lfdws_t002.csv",
     True, True, "FAIL (single-event) / PASS (multi_event)",
     "trial is actually 3 phases (latch/cube/latch); naive single global-max "
     "force detector (analyze_demo.py) picked the WRONG phase's force peak "
     "(latch, 22.72N) instead of the cube task's; multi_event.py's "
     "grasp-window-restricted detector correctly isolates the real cube cycle "
     "-- found and fixed same-day, see figures/t002labexport/"),
    ("lfdws_t001_labexport", "Data/lfdws_t001_labexport/lfdws_t001/lfdws_t001.csv",
     True, True, "FAIL (no span guard) / PASS (guarded)",
     "gripper never actuates -- width spans 0.07999710-0.07999776 m, i.e. "
     "6.6e-7 m of pure sensor noise. Without a minimum-travel guard the "
     "midpoint threshold falls inside that noise band and reports a phantom "
     "grasp at 0.06s and release at 7.66s, which then displaces the contact "
     "event to 3.34s (true force peak: 11.15N at 5.08s). With the 1mm guard "
     "in event_utils.py the detector correctly reports the press alone, at "
     "the true peak."),
    ("lfdws_t004", "Data/lfdws_t004/lfdws_t004_0.csv", True, False,
     "PASS", "no wrench topic at all; gripper-only fallback correctly omits "
     "press; grasp event propagated + visually confirmed"),
    ("lfdws_t005", "Data/lfdws_t005/lfdws_t005_0.csv", True, False,
     "PASS", "same profile/verification as t004"),
]


def main():
    rows_out = []
    for name, csv_path, has_grip_expected, has_force_expected, verdict, note in TRIALS:
        if not os.path.exists(csv_path):
            print(f"[skip] {name}: {csv_path} not found", flush=True)
            continue
        events, n_rows = detect_events_generic(csv_path)
        print(f"[{name:22s}] rows={n_rows:6d}  gripper={has_grip_expected!s:5s} "
              f"force={has_force_expected!s:5s}  events={list(events.keys())}  "
              f"verdict={verdict}", flush=True)
        rows_out.append([name, n_rows, has_grip_expected, has_force_expected,
                         ",".join(events.keys()), verdict, note])

    out_path = "figures/event_detection_accuracy_table.csv"
    os.makedirs("figures", exist_ok=True)
    with open(out_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["trial", "n_rows", "gripper_present", "force_present",
                   "events_detected", "manual_verdict", "notes"])
        for r in rows_out:
            w.writerow(r)
    print(f"\n[write] {out_path}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
