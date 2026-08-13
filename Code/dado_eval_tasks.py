"""
The shared evaluation set for label-free-baseline comparisons.

Both label-free baselines must be scored on exactly the same (recording,
event) pairs against exactly the same ground truth, or the comparison is
meaningless. This module owns that list so neither script can drift from
the other:

    Code/_dado_vs_groundtruth_all_trials.py   DINOv2 attention x real depth
    Code/baseline_sam_depth_ranking.py        SAM automasks ranked by depth

It was pulled out of the DADO script when the second baseline needed the
same list: importing that module to reuse it dragged in `transformers`,
which only exists in .venv_dado, so the .venv_sam2 baseline could not load
it. This module deliberately depends on nothing beyond csv/numpy/os plus
event_utils, so it imports cleanly in every venv.

Each entry is:
    (trial, event_name, role, color_bgr, rgb_path, depth_path,
     sidecar_summary_csv, rgb_img_id)

Event selection uses the same guarded detector as the rest of the pipeline
(see event_utils.py) -- notably lfdws_t001_labexport's gripper never
actuates, so it contributes its contact event only, not the phantom
grasp/release an unguarded midpoint threshold would invent.
"""
import ast
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from event_utils import gripper_closed_window, gripper_transitions

GRIP = "NS_1.franka_gripper.joint_states.position"
FX = "bota_post.wrench_body_compensated.wrench.force.x"
FY = "bota_post.wrench_body_compensated.wrench.force.y"
FZ = "bota_post.wrench_body_compensated.wrench.force.z"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"
DEPTH_COL = "zed.zed_node.depth.depth_registered.compressedDepth"


def parse_gw(c):
    try:
        return float(np.sum(ast.literal_eval(c)))
    except Exception:
        return float("nan")


def detect_events_generic(csv_path):
    """{event_name: (rgb_img_id, depth_img_id)}. Pure csv/numpy so it works
    in every venv. depth_img_id is the row-matched depth frame, which is not
    the same id as the rgb frame except in the lab's native export."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        rows = list(reader)
    has_force = FX in cols
    has_grip = GRIP in cols
    has_depth_col = DEPTH_COL in cols

    def ids(i):
        rgb_id = str(rows[i][IMG])
        return (rgb_id, str(rows[i][DEPTH_COL]) if has_depth_col else rgb_id)

    out = {}
    if has_force:
        fm = np.array([np.sqrt(float(r[FX])**2 + float(r[FY])**2 + float(r[FZ])**2)
                       for r in rows])
        baseline = np.median(fm[:len(fm)//10])
    if not has_grip:
        if has_force:
            out["press"] = ids(int(np.argmax(fm - baseline)))
        return out
    w = np.array([parse_gw(r[GRIP]) for r in rows])
    grasp_i, release_i = gripper_transitions(w)
    closed = gripper_closed_window(w)
    if grasp_i is not None:
        out["grasp"] = ids(grasp_i)
    if release_i is not None:
        out["release"] = ids(release_i)
    if has_force:
        fm_adj = (np.where(closed, fm - baseline, -np.inf)
                  if closed.any() else fm - baseline)
        out["press"] = ids(int(np.argmax(fm_adj)))
    return out


def build_tasks():
    tasks = []

    # t001_depth: 5 hand-picked activity-cluster events
    # (from Code/_dado_real_depth_multi.py's event list)
    rgb = "Data/lfdws_t001_depth/zed_zed_node_rgb_color_rect_image_compressed"
    dep = "Data/lfdws_t001_depth/zed_zed_node_depth_depth_registered_compressedDepth"
    side = "figures/identify_depth_multi/objects_summary.csv"
    for name, rgb_id, depth_id, role, color in [
        ("plate_press",         "1782835513207923681", "1782835513163533696", "contact_receiver", (255, 0, 255)),
        ("screwdriver_contact", "1782835527086969733", "1782835527053175454", "tool_contact",     (0, 165, 255)),
        ("charger_grasp",       "1782835537622884551", "1782835537581977576", "charger_contact",  (0, 215, 255)),
        ("charger_lift",        "1782835540835169730", "1782835540802038251", "charger_contact",  (0, 215, 255)),
        ("charger_dock",        "1782835545525039497", "1782835545485007420", "charger_contact",  (0, 215, 255)),
    ]:
        tasks.append(("lfdws_t001_depth", name, role, color,
                      f"{rgb}/{rgb_id}.png", f"{dep}/{depth_id}.png", side, rgb_id))

    # t002_new: cube, own event detection
    rgb = "Data/lfdws_t002_new/zed_zed_node_rgb_color_rect_image_compressed"
    dep = "Data/lfdws_t002_new/zed_zed_node_depth_depth_registered_compressedDepth"
    side = "figures/t002new/identify/objects_summary.csv"
    for name, (rgb_id, depth_id) in detect_events_generic(
            "Data/lfdws_t002_new/lfdws_t002_new_0.csv").items():
        tasks.append(("lfdws_t002_new", name, "grasped", (0, 255, 0),
                      f"{rgb}/{rgb_id}.png", f"{dep}/{depth_id}.png", side, rgb_id))

    # t001_labexport: latch handle (npy depth alongside png)
    rgb = "Data/lfdws_t001_labexport/lfdws_t001/zed_zed_node_rgb_color_rect_image_compressed"
    dep = "Data/lfdws_t001_labexport/lfdws_t001/zed_zed_node_depth_depth_registered_compressedDepth"
    side = "figures/t001labexport/identify/objects_summary.csv"
    for name, (rgb_id, depth_id) in detect_events_generic(
            "Data/lfdws_t001_labexport/lfdws_t001/lfdws_t001.csv").items():
        tasks.append(("lfdws_t001_labexport", name, "contact_receiver", (255, 0, 255),
                      f"{rgb}/{rgb_id}.png", f"{dep}/{depth_id}.npy", side, rgb_id))

    # t004 / t005: gripper-only (no F/T), grasp event only -- with no force
    # there is no contact event, and release has no distinct receiving object
    for trial in ["lfdws_t004", "lfdws_t005"]:
        csvp = f"Data/{trial}/{trial}_0.csv"
        if not os.path.exists(csvp):
            csvp = next((os.path.join(f"Data/{trial}", f)
                         for f in os.listdir(f"Data/{trial}") if f.endswith(".csv")), None)
        rgb = f"Data/{trial}/zed_zed_node_rgb_color_rect_image_compressed"
        dep = f"Data/{trial}/zed_zed_node_depth_depth_registered_compressedDepth"
        side = f"figures/{trial.replace('lfdws_', '')}/identify/objects_summary.csv"
        for name, (rgb_id, depth_id) in detect_events_generic(csvp).items():
            if name != "grasp":
                continue
            tasks.append((trial, name, "grasped", (0, 255, 0),
                          f"{rgb}/{rgb_id}.png", f"{dep}/{depth_id}.png", side, rgb_id))

    return tasks


TASKS = build_tasks()
