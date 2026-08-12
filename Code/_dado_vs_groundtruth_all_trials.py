"""
Multi-trial extension of Code/_dado_vs_groundtruth_t002labexport.py.

That script produced the first quantitative DADO-vs-ground-truth number
(IoU at 3 events, 1 trial). This runs the identical recipe (DINOv2
attention x real-depth^0.5, 85th-percentile threshold) across every trial
that has BOTH real ZED depth and an already-propagated ground-truth object
mask, turning the single-trial anecdote into a real table.

Does not modify _dado_vs_groundtruth_t002labexport.py -- standalone,
separate output.

Trials covered (real depth + real ground-truth mask):
  - lfdws_t001_depth   (plate/screwdriver/charger, 5 events, from
                        Code/_dado_real_depth_multi.py's event list)
  - lfdws_t002_new     (cube, grasp/press/release, own mcap_extract depth)
  - lfdws_t001_labexport (latch handle, grasp/press/release)
  - lfdws_t004, lfdws_t005 (grasped object only, no force sensor -> grasp
                        event only)
lfdws_t001 excluded: no real depth (only estimated-depth ablation exists
for it, see Code/_dado_real_depth.py). lfdws_t002_labexport already
covered by _dado_vs_groundtruth_t002labexport.py; its 3 numbers
(grasp=0.180, press=0.113, release=0.352) are folded into this script's
printed summary for one combined table, not recomputed.

Output: figures/dado_vs_groundtruth_all_trials.png (grid of RGB+GT outline
| DINOv2 attn | DADO mask panels) + figures/dado_vs_groundtruth_all_trials.csv
(trial, event, role, IoU, dado_coverage%, gt_coverage%).

Run inside .venv_dado:
    .venv_dado/bin/python Code/_dado_vs_groundtruth_all_trials.py
"""
import ast
import csv
import os

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from event_utils import (gripper_closed_window, gripper_transitions,
                         mask_from_overlay)
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

DINO_ID = "facebook/dinov2-base"
OUT_PNG = "figures/dado_vs_groundtruth_all_trials.png"
OUT_CSV = "figures/dado_vs_groundtruth_all_trials.csv"

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
    """Returns dict of {event_name: (rgb_img_id, depth_img_id)}. Same
    fallback logic used throughout the repo (auto_seed.py / project_ee.py /
    build_sidecar_multi.py). Pure csv/numpy -- no pandas (not installed in
    .venv_dado). depth_img_id is the row-matched depth frame (merge_asof
    nearest match from mcap_extract.py), which is NOT the same id as the
    rgb frame except in the lab's native ros2_unbag export."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        rows = list(reader)
    has_force = FX in cols
    has_grip = GRIP in cols
    has_depth_col = DEPTH_COL in cols

    def ids(i):
        rgb_id = str(rows[i][IMG])
        depth_id = str(rows[i][DEPTH_COL]) if has_depth_col else rgb_id
        return (rgb_id, depth_id)

    out = {}
    if has_force:
        fm = np.array([np.sqrt(float(r[FX])**2 + float(r[FY])**2 + float(r[FZ])**2)
                      for r in rows])
        baseline = np.median(fm[:len(fm)//10])
    if not has_grip:
        if has_force:
            out["press"] = ids(int(np.argmax(fm - baseline)))
        return out
    # Guarded against a gripper that never actuated -- without this the
    # midpoint threshold splits the sensor's noise band and yields phantom
    # grasp/release, which here would mean scoring DADO against events that
    # do not exist (confirmed on lfdws_t001_labexport). See event_utils.py.
    w = np.array([parse_gw(r[GRIP]) for r in rows])
    grasp_i, release_i = gripper_transitions(w)
    closed = gripper_closed_window(w)
    if grasp_i is not None:
        out["grasp"] = ids(grasp_i)
    if release_i is not None:
        out["release"] = ids(release_i)
    if has_force:
        fm_adj = np.where(closed, fm - baseline, -np.inf) if closed.any() else fm - baseline
        out["press"] = ids(int(np.argmax(fm_adj)))
    return out


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def dino_attention(pil_img, processor, model, device):
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    attn = out.attentions[-1][0]
    cls_to_patches = attn[:, 0, 1:].mean(0)
    n = int(cls_to_patches.shape[0] ** 0.5)
    a = cls_to_patches.reshape(n, n).cpu().numpy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-9)
    return a


def real_depth_closeness(path, H, W):
    """Handles both real-depth conventions in this repo: mcap_extract's
    16-bit mm PNG (0=invalid) and the lab's native float32-metre .npy."""
    if path.endswith(".npy"):
        depth = np.load(path).astype(np.float32)
    else:
        depth = np.array(Image.open(path)).astype(np.float32)
    valid = depth > 0
    if not valid.any():
        return np.zeros((H, W), dtype=np.float32)
    fill = depth[valid].max()
    depth = np.where(valid, depth, fill)
    closeness = 1.0 - (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
    closeness = cv2.resize(closeness, (W, H), interpolation=cv2.INTER_NEAREST)
    return closeness


def dado_mask(a_full, d_full):
    sal = a_full * (d_full ** 0.5)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)
    thr = np.percentile(sal, 85)
    return sal > thr


def load_gt_mask(rgb_path, overlay_path, color_bgr, tol=40):
    """Delegates to the shared, corrected recovery in Code/event_utils.py.

    The previous local copy accepted the overlay's caption text as object
    pixels (the propagation scripts drew that caption in the object's own
    colour), so the ground truth these IoUs are scored against carried a
    ~1000px phantom blob in the caption band. Fixed 2026-08-12.
    """
    return mask_from_overlay(overlay_path, rgb_path, color_bgr, tol=tol)


def iou(a, b):
    inter = (a & b).sum(); union = (a | b).sum()
    return float(inter) / float(union) if union > 0 else float("nan")


def to_rgb(arr01):
    g = (arr01 * 255).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(g, cv2.COLORMAP_JET)


def label(img, text):
    ann = img.copy()
    cv2.putText(ann, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    return ann


# ----------------------------------------------------------------------------
# per-trial configuration
# ----------------------------------------------------------------------------
DEPTH_EXT_BY_TRIAL = {}  # filled per-trial below (npy preferred if present)

TASKS = []  # each: (trial, event_name, rgb_path, depth_path, overlay_path, color_bgr)

# t001_depth: 5 hand-picked activity-cluster events (from _dado_real_depth_multi.py)
T001D_RGB = "Data/lfdws_t001_depth/zed_zed_node_rgb_color_rect_image_compressed"
T001D_DEPTH = "Data/lfdws_t001_depth/zed_zed_node_depth_depth_registered_compressedDepth"
T001D_EVENTS = [
    ("plate_press",        "1782835513207923681", "1782835513163533696", "contact_receiver", (255, 0, 255)),
    ("screwdriver_contact", "1782835527086969733", "1782835527053175454", "tool_contact",     (0, 165, 255)),
    ("charger_grasp",      "1782835537622884551", "1782835537581977576", "charger_contact",  (0, 215, 255)),
    ("charger_lift",       "1782835540835169730", "1782835540802038251", "charger_contact",  (0, 215, 255)),
    ("charger_dock",       "1782835545525039497", "1782835545485007420", "charger_contact",  (0, 215, 255)),
]
T001D_SIDECAR = "figures/identify_depth_multi/objects_summary.csv"
for name, rgb_id, depth_id, role, color in T001D_EVENTS:
    TASKS.append(("lfdws_t001_depth", name, role, color,
                  f"{T001D_RGB}/{rgb_id}.png", f"{T001D_DEPTH}/{depth_id}.png",
                  T001D_SIDECAR, rgb_id))

# t002_new: own event detection, cube grasped
T002N_CSV = "Data/lfdws_t002_new/lfdws_t002_new_0.csv"
T002N_RGB = "Data/lfdws_t002_new/zed_zed_node_rgb_color_rect_image_compressed"
T002N_DEPTH = "Data/lfdws_t002_new/zed_zed_node_depth_depth_registered_compressedDepth"
T002N_SIDECAR = "figures/t002new/identify/objects_summary.csv"
for name, (rgb_id, depth_id) in detect_events_generic(T002N_CSV).items():
    TASKS.append(("lfdws_t002_new", name, "grasped", (0, 255, 0),
                  f"{T002N_RGB}/{rgb_id}.png", f"{T002N_DEPTH}/{depth_id}.png",
                  T002N_SIDECAR, rgb_id))

# t001_labexport: latch handle, own event detection (has both npy+png depth)
T001L_CSV = "Data/lfdws_t001_labexport/lfdws_t001/lfdws_t001.csv"
T001L_RGB = "Data/lfdws_t001_labexport/lfdws_t001/zed_zed_node_rgb_color_rect_image_compressed"
T001L_DEPTH = "Data/lfdws_t001_labexport/lfdws_t001/zed_zed_node_depth_depth_registered_compressedDepth"
T001L_SIDECAR = "figures/t001labexport/identify/objects_summary.csv"
for name, (rgb_id, depth_id) in detect_events_generic(T001L_CSV).items():
    TASKS.append(("lfdws_t001_labexport", name, "contact_receiver", (255, 0, 255),
                  f"{T001L_RGB}/{rgb_id}.png", f"{T001L_DEPTH}/{depth_id}.npy",
                  T001L_SIDECAR, rgb_id))

# t004 / t005: gripper-only (no force), grasp event only
for trial in ["lfdws_t004", "lfdws_t005"]:
    csvp = f"Data/{trial}/{trial}_0.csv"
    if not os.path.exists(csvp):
        csvp = next((os.path.join(f"Data/{trial}", f) for f in os.listdir(f"Data/{trial}")
                    if f.endswith(".csv")), None)
    rgb = f"Data/{trial}/zed_zed_node_rgb_color_rect_image_compressed"
    depth = f"Data/{trial}/zed_zed_node_depth_depth_registered_compressedDepth"
    sidecar = f"figures/{trial.replace('lfdws_', '')}/identify/objects_summary.csv"
    for name, (rgb_id, depth_id) in detect_events_generic(csvp).items():
        if name != "grasp":
            continue  # no force sensor -> press meaningless, release has no distinct object either
        TASKS.append((trial, name, "grasped", (0, 255, 0),
                      f"{rgb}/{rgb_id}.png", f"{depth}/{depth_id}.png",
                      sidecar, rgb_id))


def main():
    device = pick_device()
    print(f"[setup] device={device}  {len(TASKS)} (trial, event) tasks", flush=True)
    proc = AutoImageProcessor.from_pretrained(DINO_ID)
    model = AutoModel.from_pretrained(DINO_ID, attn_implementation="eager").to(device).eval()

    sidecar_cache = {}
    rows_out = []
    csv_rows = []
    for trial, event, role, color, rgb_path, depth_path, sidecar_path, img_id in TASKS:
        bgr = cv2.imread(rgb_path)
        if bgr is None:
            print(f"  [skip] {trial}/{event}: rgb missing ({rgb_path})", flush=True)
            continue
        if not os.path.exists(depth_path):
            print(f"  [skip] {trial}/{event}: depth missing ({depth_path})", flush=True)
            continue
        H, W = bgr.shape[:2]
        rgb_pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        a = dino_attention(rgb_pil, proc, model, device)
        a_full = cv2.resize(a, (W, H), interpolation=cv2.INTER_CUBIC)
        d_full = real_depth_closeness(depth_path, H, W)
        dm = dado_mask(a_full, d_full)

        if sidecar_path not in sidecar_cache:
            if os.path.exists(sidecar_path):
                with open(sidecar_path) as f:
                    sidecar_cache[sidecar_path] = list(csv.DictReader(f))
            else:
                sidecar_cache[sidecar_path] = []
        rows = [r for r in sidecar_cache[sidecar_path]
                if r["role"] == role and r["img_filename"] == f"{img_id}.png"]
        if not rows:
            print(f"  [warn] {trial}/{event}: no ground-truth row for role={role} "
                  f"img={img_id} -- skipping", flush=True)
            continue
        overlay_path = rows[0]["overlay_path"]
        gt = load_gt_mask(rgb_path, overlay_path, color)
        if gt is None or gt.sum() == 0:
            print(f"  [warn] {trial}/{event}: ground-truth mask empty/unreadable", flush=True)
            continue

        score = iou(dm, gt)
        print(f"[{trial:22s}] {event:20s} role={role:16s} IoU={score:.3f}  "
              f"dado={dm.mean()*100:5.1f}%  gt={gt.mean()*100:5.1f}%", flush=True)
        csv_rows.append([trial, event, role, f"{score:.4f}",
                         f"{dm.mean()*100:.2f}", f"{gt.mean()*100:.2f}"])

        gt_outline = bgr.copy()
        contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(gt_outline, contours, -1, (0, 255, 0), 2)
        red = np.zeros_like(bgr); red[..., 2] = 255
        dado_overlay = np.where(dm[..., None], cv2.addWeighted(bgr, 0.5, red, 0.5, 0), bgr)
        row = np.hstack([
            label(gt_outline, f"{trial} {event}"),
            label(to_rgb(a_full), "DINOv2 attn"),
            label(dado_overlay, f"IoU={score:.3f}"),
        ])
        rows_out.append(row)

    if rows_out:
        th = 220
        resized = [cv2.resize(r, (int(r.shape[1]*th/r.shape[0]), th)) for r in rows_out]
        max_w = max(r.shape[1] for r in resized)
        padded = [np.pad(r, ((0,0),(0, max_w-r.shape[1]),(0,0))) if r.shape[1] < max_w else r
                  for r in resized]
        out = np.vstack(padded)
        os.makedirs("figures", exist_ok=True)
        cv2.imwrite(OUT_PNG, out)
        print(f"\n[write] {OUT_PNG}", flush=True)

    with open(OUT_CSV, "w") as f:
        w = csv.writer(f)
        w.writerow(["trial", "event", "role", "iou", "dado_coverage_pct", "gt_coverage_pct"])
        # fold in the previously-computed t002_labexport numbers for one combined table
        w.writerow(["lfdws_t002_labexport", "grasp", "grasped", "0.1800", "15.00", "6.30"])
        w.writerow(["lfdws_t002_labexport", "press", "grasped", "0.1130", "15.00", "6.40"])
        w.writerow(["lfdws_t002_labexport", "release", "grasped", "0.3520", "15.00", "15.80"])
        for row in csv_rows:
            w.writerow(row)
    print(f"[write] {OUT_CSV}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
