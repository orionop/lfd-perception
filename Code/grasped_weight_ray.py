"""
Seed the GRASPED role from the held object's own weight, with no gripper
offset.

WHY THIS EXISTS
---------------
Every previous attempt to seed the grasped role needed a bota -> gripper
centre vector, and all of them failed:
  Code/derive_tool_axis.py      0/1440 angles in the constrained family
  Code/measure_grasp_offset.py  LOTO 9/17
  Code/grasp_offset_search.py   16/17 in fold, but the basin gate rejected it
  Code/pose_frame_probe.py      all six signed 104 mm placements 0/17
That last one says the 104 mm from Mark's drawing does not describe the
current_pose frame, and until the lab says which link current_pose publishes,
any offset based approach is building on an unknown.

THE IDEA THAT SIDESTEPS IT
--------------------------
bota_post/wrench_body_compensated is already compensated for the mass of the
tool itself. So while an object is held and the arm is not accelerating hard,
the residual wrench is that OBJECT's weight, and by Bicchi 1990 the line of
action of a pure force passes through the point where it acts, here the held
object's centre of mass. Projecting that line therefore sweeps THROUGH the
grasped object, exactly as the contact ray sweeps through the contact
receiver.

No gripper offset appears anywhere. The only calibration used is
T_bota_camera, which is independently validated at 6/7 contact events.

WHAT IS ACTUALLY BEING ASSUMED, AND HOW IT CAN FAIL
---------------------------------------------------
1. Body compensation is good enough that the residual is the payload, not
   uncompensated tool mass. If it is not, the ray points at a fixed phantom.
   The static-arm requirement below is what keeps inertial terms out.
2. A held object weighs enough to give a well conditioned line. A 100 g part
   is about 1 N, and r0 = (f x tau)/|f|^2 divides by |f|^2, so small forces
   put r0 far away and noisy. Samples below MIN_HOLD_FORCE_N are skipped and
   counted, not silently dropped.
3. A two finger squeeze is internal and nets to zero at the wrist, so this
   sees the weight and not the grip. That is the point, but it also means
   this cannot work before the object leaves its support.

THE CONTROLS, BECAUSE A RAY THAT HITS IS NOT AUTOMATICALLY A RESULT
-------------------------------------------------------------------
The grasped mask can be large, so a ray crossing the image has a real chance
of hitting it by accident. Two nulls are reported alongside the hit rate:
  shuffled  the wrench of one sample paired with the pose of another, which
            destroys the physical correspondence while keeping both marginal
            distributions intact
  vertical  a plain downward line through the current_pose origin, which is
            what you would get from assuming the object hangs straight below
            without using the torque at all
If the real ray does not beat both, this idea is not carrying information.

Read only. Writes figures/grasped_weight_ray.csv and prints per sample.

Usage: .venv_analysis/bin/python Code/grasped_weight_ray.py
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from event_utils import gripper_closed_window, parse_gripper_width
from geometric_seed import load_calib
from wrench_ray import pose_to_T, ray_mask_score, ray_pixels, wrench_line_bota

OUT_CSV = "figures/grasped_weight_ray.csv"

PX = "NS_1.franka_robot_state_broadcaster.current_pose.pose.position."
QX = "NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation."
FX = "bota_post.wrench_body_compensated.wrench.force."
TX = "bota_post.wrench_body_compensated.wrench.torque."
GRIP = "NS_1.franka_gripper.joint_states.position"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"

# Below this the line of action is too ill conditioned to trust (r0 divides
# by |f|^2). 1.5 N is roughly a 150 g payload.
MIN_HOLD_FORCE_N = 1.5

# Skip the first and last part of the held window: the object is still
# supported early on, and being set down late.
HOLD_TRIM_FRAC = 0.25

# How many frames to sample inside the trimmed hold.
N_SAMPLES = 8

# (label, merged csv, objects_summary.csv)
TRIALS = [
    ("lfdws_t001", "Data/lfdws_t001/lfdws_t001/lfdws_t001_0.csv",
     "figures/identify/objects_summary.csv"),
    ("lfdws_t002_new", "Data/lfdws_t002_new/lfdws_t002_new_0.csv",
     "figures/t002new/identify/objects_summary.csv"),
    ("lfdws_t002_labexport", "Data/lfdws_t002_labexport/lfdws_t002/lfdws_t002.csv",
     "figures/t002labexport/identify/objects_summary.csv"),
]


def load_masks(sidecar, role="grasped"):
    """{img_id: (x0,y0,x1,y1)} for frames where the role is present."""
    out = {}
    if not os.path.exists(sidecar):
        return out
    with open(sidecar) as f:
        for row in csv.DictReader(f):
            if row.get("role") != role:
                continue
            try:
                px = int(float(row["mask_px"]))
                bb = (float(row["bbox_x0"]), float(row["bbox_y0"]),
                      float(row["bbox_x1"]), float(row["bbox_y1"]))
            except (KeyError, ValueError):
                continue
            if px <= 0 or bb[0] < 0:
                continue
            out[os.path.splitext(row["img_filename"])[0]] = bb
    return out


def bbox_mask(bb, W, H):
    """Boolean mask from a bbox. The sidecar stores bboxes, not masks, so a
    hit here means 'inside the object's bounding box' -- a weaker claim than
    'inside the mask', and the printout says so."""
    m = np.zeros((H, W), dtype=bool)
    x0, y0, x1, y1 = [int(round(v)) for v in bb]
    m[max(0, y0):min(H, y1 + 1), max(0, x0):min(W, x1 + 1)] = True
    return m


def collect(label, merged, sidecar, K, T_bc):
    if not os.path.exists(merged):
        print(f"  [skip] merged CSV missing: {merged}", flush=True)
        return []
    boxes = load_masks(sidecar)
    if not boxes:
        print(f"  [skip] no grasped rows in {sidecar}", flush=True)
        return []
    with open(merged) as f:
        rows = list(csv.DictReader(f))
    if GRIP not in rows[0] or (FX + "x") not in rows[0]:
        print("  [skip] needs both gripper and wrench topics", flush=True)
        return []

    w = np.array([parse_gripper_width(r[GRIP]) for r in rows])
    closed = gripper_closed_window(w)
    if not closed.any():
        print("  [skip] gripper never actuated", flush=True)
        return []
    idx = np.flatnonzero(closed)
    lo, hi = idx[0], idx[-1]
    span = hi - lo
    a = int(lo + HOLD_TRIM_FRAC * span)
    b = int(hi - HOLD_TRIM_FRAC * span)
    print(f"  [hold] rows {lo}..{hi}, trimmed to {a}..{b}", flush=True)

    # one row per distinct image frame inside the trimmed hold
    seen, cand = set(), []
    for i in range(a, b + 1):
        iid = str(rows[i][IMG])
        if iid in seen or iid not in boxes:
            continue
        seen.add(iid)
        cand.append(i)
    if not cand:
        print("  [skip] no tracked frames inside the trimmed hold", flush=True)
        return []
    pick = np.linspace(0, len(cand) - 1, min(N_SAMPLES, len(cand)))
    out, weak = [], 0
    for j in np.unique(pick.astype(int)):
        i = cand[j]
        r = rows[i]
        f = np.array([float(r[FX + c]) for c in "xyz"])
        tau = np.array([float(r[TX + c]) for c in "xyz"])
        if np.linalg.norm(f) < MIN_HOLD_FORCE_N:
            weak += 1
            continue
        pose = tuple(float(r[PX + c]) for c in "xyz") + \
               tuple(float(r[QX + c]) for c in "xyzw")
        out.append({"trial": label, "img_id": str(r[IMG]), "f": f,
                    "tau": tau, "pose": pose, "bbox": boxes[str(r[IMG])]})
    print(f"  [collect] {len(out)} samples ({weak} below "
          f"{MIN_HOLD_FORCE_N} N, skipped)", flush=True)
    return out


def score(samples, K, T_bc, W, H, mode="real", rng=None):
    hits, dists = 0, []
    n = len(samples)
    for i, s in enumerate(samples):
        if mode == "shuffled":
            src = samples[(i + 1 + (rng.integers(n - 1) if n > 2 else 0)) % n]
            f, tau = src["f"], src["tau"]
        else:
            f, tau = s["f"], s["tau"]
        T_bb = pose_to_T(*s["pose"])
        m = bbox_mask(s["bbox"], W, H)
        if mode == "vertical":
            # straight down through the pose origin, torque unused
            R = T_bb[:3, :3]
            fhat = R.T @ np.array([0.0, 0.0, -1.0])
            r0 = np.zeros(3)
        else:
            r0, fhat = wrench_line_bota(f, tau)
            if r0 is None:
                continue
        uv, _ = ray_pixels(r0, fhat, T_bb, T_bc, K, s_min=-0.6, s_max=0.6,
                           n=240)
        hit, d, n_in = ray_mask_score(uv, m, W, H)
        hits += int(hit)
        if np.isfinite(d):
            dists.append(d)
        if mode == "real":
            print(f"    [{s['trial']:22s} {s['img_id'][:10]}] |F|={np.linalg.norm(f):6.2f} N  "
                  f"in-frame={n_in:4d}  {'HIT ' if hit else 'MISS'}  "
                  f"d={d:8.1f} px", flush=True)
    return hits, len(samples), (float(np.mean(dists)) if dists else float("nan"))


def main():
    K, T_bc = load_calib()
    W = int(K[0, 2] * 2)
    H = int(K[1, 2] * 2)
    print(f"[calib] image {W}x{H}, bota_to_camera t = "
          f"{(T_bc[:3, 3] * 1000).round(2).tolist()} mm\n", flush=True)

    samples = []
    for label, merged, sidecar in TRIALS:
        print(f"[trial] {label}", flush=True)
        samples += collect(label, merged, sidecar, K, T_bc)
    if not samples:
        print("[fatal] no usable held samples", flush=True)
        return

    print(f"\n[data] {len(samples)} held samples over "
          f"{len(set(s['trial'] for s in samples))} recordings", flush=True)
    print("[note] scoring against the grasped BOUNDING BOX, not the mask, "
          "since\n       the sidecar stores bboxes. A hit is therefore a "
          "weaker claim.\n", flush=True)

    rng = np.random.default_rng(0)
    h, n, d = score(samples, K, T_bc, W, H, "real")
    print(f"\n[real]      {h}/{n} = {h / n:.3f}   mean dist {d:.1f} px",
          flush=True)
    hs, _, ds = score(samples, K, T_bc, W, H, "shuffled", rng)
    print(f"[shuffled]  {hs}/{n} = {hs / n:.3f}   mean dist {ds:.1f} px",
          flush=True)
    hv, _, dv = score(samples, K, T_bc, W, H, "vertical")
    print(f"[vertical]  {hv}/{n} = {hv / n:.3f}   mean dist {dv:.1f} px",
          flush=True)

    print("\n[verdict]", flush=True)
    if h > hs and h > hv:
        print("  the weight ray beats BOTH nulls -- worth pursuing as a "
              "grasped seed.", flush=True)
    elif h <= hs:
        print("  does NOT beat the shuffled null: the apparent hits do not "
              "depend on\n  the wrench actually belonging to that pose. "
              "Not a signal.", flush=True)
    else:
        print("  does not beat a plain vertical line, so the torque is "
              "adding nothing\n  over 'assume it hangs below the wrist'.",
              flush=True)

    os.makedirs("figures", exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fo:
        wtr = csv.writer(fo)
        wtr.writerow(["mode", "hits", "n", "hit_rate", "mean_dist_px"])
        for name, hh, dd in [("real", h, d), ("shuffled", hs, ds),
                             ("vertical", hv, dv)]:
            wtr.writerow([name, hh, n, f"{hh / n:.4f}", f"{dd:.2f}"])
    print(f"[write] {OUT_CSV}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
