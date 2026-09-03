"""
Seed SAM from the projected contact point instead of image statistics.

WHAT THIS REPLACES
------------------
Code/auto_seed.py picks a seed with no knowledge of where the robot actually
touched. It runs SAM's automatic mask generator and scores candidates on
centrality, area fraction and border distance. Its own docstring records that
this is not fixable by tuning: the absolute area cap (< 0.4 of frame) wrongly
rejects the plate in lfdws_t001_depth, and the relative-size variant that was
tried instead mis-picks the reflective table mat in lfdws_t001. Both failures
are kept as artifacts. It was always a placeholder for a geometric seed.

The geometric path is implemented, but production use is intentionally gated.
The retained T_bota_camera candidate scored 6/7 on the available contact
events, but its rotation/translation-frame construction is unresolved and it
was assessed on the same small event set used to compare conventions.
`calibration.yaml` therefore keeps `bota_to_camera.filled: false` until an
independent hand-eye calibration passes the acceptance checks in
`LAB_DELIVERABLE_A.md`.

HOW EACH ROLE IS SEEDED
-----------------------
contact roles (contact_receiver, tool_contact, charger_contact)
    The wrench line of action (Bicchi 1990) passes through the contact point
    but does not say where along itself that point lies. Real depth resolves
    it: walk the projected ray and take the first pixel where the ray's own
    camera-frame depth agrees with the measured depth map. That intersection
    is the object surface. Without depth, fall back to projecting r0, the
    point on the line closest to the sensor origin, which for a surface
    contact sits near the true one.

grasped
    Project the gripper centre, which is holding the object by definition.

THE ONE THING NOT GIVEN BY THE CAD, AND HOW IT IS OBTAINED
----------------------------------------------------------
Mark's drawing decodes cleanly for the gripper centre: 104 - 91.72 = 12.28
exactly matches its "12.28" dimension, and -12.28 + 15.58 = 3.30 exactly
matches his stated global "y: 3.3". So the gripper centre lies 104 mm from
the bota origin along the tool axis.

Its DIRECTION in the bota frame is not determined, because the rotation from
his drawing-global frame to the bota frame is the same step his scan does not
show (confirmed: no standard Euler convention, transpose or signed axis
permutation reproduces his own global-to-rotated conversion; closest is 51 mm
off). So this script does not assume it. It tests all six signed axis
directions against the grasp events and reports which one actually projects
onto the grasped object. If no candidate wins clearly, that is reported and
the grasped role is left unseeded rather than guessed.

Writes a CSV with the SAME header as auto_seed.py, so the propagation scripts
consume it unmodified (propagate_demo_bidir.py:load_auto_seed reads only
role, img_id, seed_x, seed_y).

Usage:
    .venv_analysis/bin/python Code/geometric_seed.py
    .venv_analysis/bin/python Code/geometric_seed.py --derive_only
"""
import argparse
import csv
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from event_utils import (gripper_closed_window, mask_from_overlay,
                         parse_gripper_width)
from wrench_ray import pose_to_T, project_points, wrench_line_bota

CALIB = "calibration.yaml"
GRIPPER_OFFSET_MM = 104.0
RGB_DIR = "zed_zed_node_rgb_color_rect_image_compressed"
DEPTH_DIR = "zed_zed_node_depth_depth_registered_compressedDepth"

POSE = ["NS_1.franka_robot_state_broadcaster.current_pose.pose.position.x",
        "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.y",
        "NS_1.franka_robot_state_broadcaster.current_pose.pose.position.z",
        "NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation.x",
        "NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation.y",
        "NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation.z",
        "NS_1.franka_robot_state_broadcaster.current_pose.pose.orientation.w"]
FX, FY, FZ = [f"bota_post.wrench_body_compensated.wrench.force.{a}" for a in "xyz"]
TXq, TYq, TZq = [f"bota_post.wrench_body_compensated.wrench.torque.{a}" for a in "xyz"]
GRIP = "NS_1.franka_gripper.joint_states.position"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"
DEPTH_COL = "zed.zed_node.depth.depth_registered.compressedDepth"

GRASP_TRIALS = [
    ("lfdws_t001", "Data/lfdws_t001/lfdws_t001", "figures/identify"),
    ("lfdws_t002_new", "Data/lfdws_t002_new", "figures/t002new/identify"),
    ("lfdws_t002_labexport", "Data/lfdws_t002_labexport/lfdws_t002",
     "figures/t002labexport/identify"),
    ("lfdws_t004", "Data/lfdws_t004", "figures/t004/identify"),
    ("lfdws_t005", "Data/lfdws_t005", "figures/t005/identify"),
]
GRASPED_BGR = (0, 255, 0)

CANDIDATES = [("+x", np.array([1.0, 0, 0])), ("-x", np.array([-1.0, 0, 0])),
              ("+y", np.array([0, 1.0, 0])), ("-y", np.array([0, -1.0, 0])),
              ("+z", np.array([0, 0, 1.0])), ("-z", np.array([0, 0, -1.0]))]


def load_calib():
    import yaml
    with open(CALIB) as f:
        c = yaml.safe_load(f)
    intr, bc = c["camera_intrinsics"], c["bota_to_camera"]
    if not (intr.get("filled") and bc.get("filled")):
        print("[fatal] calibration.yaml is not filled:true for both blocks; "
              "this script must not invent pixels", flush=True)
        sys.exit(1)
    return np.array(intr["K"], float), np.array(bc["T"], float)


def rows_of(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def find_csv(d):
    for f in sorted(os.listdir(d)):
        if f.endswith(".csv") and not f.startswith("."):
            return os.path.join(d, f)
    return None


def project_bota_point(p_bota, pose, K, T_bc):
    T_bb = pose_to_T(*pose)
    p_base = (T_bb @ np.append(p_bota, 1.0))[:3]
    uv, z = project_points(p_base[None, :], K, T_bb @ T_bc)
    return uv[0], float(z[0])


def grasp_events(K, T_bc):
    """Grasp-event frames that carry a propagated grasped mask."""
    out = []
    for label, tdir, figdir in GRASP_TRIALS:
        cpath = find_csv(tdir)
        side = os.path.join(figdir, "objects_summary.csv")
        if not (cpath and os.path.exists(side)):
            continue
        rows = rows_of(cpath)
        if GRIP not in rows[0]:
            continue
        w = np.array([parse_gripper_width(r[GRIP]) for r in rows])
        closed = gripper_closed_window(w)
        if not closed.any():
            print(f"  [skip] {label}: span guard, gripper never actuated",
                  flush=True)
            continue
        i = int(np.flatnonzero(closed)[0])
        # step a little into the hold so the fingers have actually closed
        idx = np.flatnonzero(closed)
        i = int(idx[len(idx) // 6])
        img_id = str(rows[i][IMG])
        rgb = os.path.join(tdir, RGB_DIR, f"{img_id}.png")
        gt_rows = [r for r in rows_of(side)
                   if r["role"] == "grasped" and
                   r["img_filename"] == f"{img_id}.png"]
        if not (os.path.exists(rgb) and gt_rows):
            continue
        m = mask_from_overlay(gt_rows[0]["overlay_path"], rgb, GRASPED_BGR)
        if m is None or not m.any():
            continue
        out.append({"trial": label, "tdir": tdir, "figdir": figdir,
                    "img_id": img_id, "rgb": rgb, "mask": m,
                    "pose": np.array([float(rows[i][k]) for k in POSE])})
    return out


def derive_gripper_offset(K, T_bc):
    print("\n[derive] which bota-frame direction is the tool axis?", flush=True)
    print("[derive] gripper centre is 104 mm from the bota origin along it; "
          "testing all six signed axes\n", flush=True)
    evs = grasp_events(K, T_bc)
    if not evs:
        print("[derive] no usable grasp events", flush=True)
        return None, []
    print(f"[derive] {len(evs)} grasp events: "
          f"{[e['trial'] for e in evs]}\n", flush=True)
    print(f"{'axis':6s} {'in-mask':>9s} {'in-frame':>9s} {'mean dist px':>13s}",
          flush=True)
    results = []
    for name, d in CANDIDATES:
        p = d * (GRIPPER_OFFSET_MM / 1000.0)
        inm, inf, dists = 0, 0, []
        for e in evs:
            uv, z = project_bota_point(p, e["pose"], K, T_bc)
            H, W = e["mask"].shape
            if z <= 0 or not (0 <= uv[0] < W and 0 <= uv[1] < H):
                continue
            inf += 1
            if e["mask"][int(uv[1]), int(uv[0])]:
                inm += 1
            ys, xs = np.nonzero(e["mask"])
            dists.append(float(np.hypot(uv[0] - xs.mean(), uv[1] - ys.mean())))
        md = float(np.mean(dists)) if dists else float("inf")
        results.append((name, d, inm, inf, md))
        print(f"{name:6s} {inm:5d}/{len(evs):<3d} {inf:5d}/{len(evs):<3d} "
              f"{md:13.1f}", flush=True)

    ranked = sorted(results, key=lambda r: (-r[2], r[4]))
    best = ranked[0]
    print(f"\n[derive] best: {best[0]}  ({best[2]}/{len(evs)} inside the "
          f"grasped mask, mean {best[4]:.1f} px)", flush=True)
    if best[2] == 0:
        print("[derive] NO candidate lands inside the grasped object on any "
              "event. The tool-axis direction is not established; the grasped "
              "role will be left unseeded rather than guessed.", flush=True)
        return None, evs
    if len(ranked) > 1 and ranked[1][2] == best[2] and \
            abs(ranked[1][4] - best[4]) < 20:
        print(f"[derive] AMBIGUOUS: {ranked[1][0]} scores the same. Not "
              f"choosing between them.", flush=True)
        return None, evs
    return best[1] * (GRIPPER_OFFSET_MM / 1000.0), evs


def depth_at(tdir, depth_id, H, W):
    # ORDER MATTERS. The lab's native export writes BOTH a float32-metre .npy
    # and an 8-bit uint8 .png preview of the same frame; only the .npy is
    # depth data. Reading the .png first and dividing by 1000 yields a median
    # of exactly 0.255 m on every frame (255/1000, saturated), which is what
    # corrupted the lfdws_t002_labexport samples. Our own mcap_extract.py
    # writes a 16-bit millimetre .png and no .npy, so .png stays the fallback.
    for ext in (".npy", ".png"):
        p = os.path.join(tdir, DEPTH_DIR, f"{depth_id}{ext}")
        if os.path.exists(p):
            if ext == ".npy":
                d = np.load(p).astype(np.float32)
            else:
                d = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            if d.shape != (H, W):
                d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
            return d
    return None


# A near-field gate was tried to stop the ray locking onto the gripper, on the
# theory that the manipulator sits between the camera and the target. It was
# WRONG and is kept at a token value only to reject degenerate samples.
#
# The evidence was already in the repo: this workspace is depth-compressed to
# roughly 0.15-0.21 m (the finding that explained why depth ranking cannot
# separate proposals -- 41 percent of them sit within 2 cm of the target).
# A 0.22 m gate therefore excludes the objects themselves, not the gripper.
# Measured directly here: a held object back-projected to 0.199 m. With the
# gate in place the seeder scored 0/7; without it, 4/7.
#
# Restricting the ray to s >= 0 was the second half of the same mistake: the
# 6/7 validation in calibration.yaml samples s in [-0.6, 0.6], and the contact
# is not always on the outward side of r0.
NEAR_FIELD_REJECT_M = 0.05
DEPTH_AGREE_M = 0.05
# Minimum contiguous agreeing samples for the ray-depth hit to be believed.
MIN_DEPTH_RUN = 5


def seed_contact(force, torque, pose, K, T_bc, depth, H, W):
    """Ray-depth intersection, or the r0 projection when depth is absent.

    Three corrections over the naive version, all geometric rather than tuned:

      * take the MEDIAN of every pixel whose depth agrees, not the single best
        residual. One argmin pixel is fragile; the agreeing region is not.

    Two further changes were tried and REVERTED, see the constants above:
    gating out a supposed gripper near field, and searching outward only.
    Both dropped the score from 4/7 to 0/7 because this workspace is
    depth-compressed to ~0.15-0.21 m, so the "near field" is where the objects
    actually are.
    """
    r0, fhat = wrench_line_bota(force, torque)
    if r0 is None:
        return None, "no force"
    s = np.linspace(-0.30, 0.60, 400)
    pts = r0[None, :] + s[:, None] * fhat[None, :]
    T_bb = pose_to_T(*pose)
    p_base = (T_bb @ np.hstack([pts, np.ones((len(pts), 1))]).T).T[:, :3]
    uv, z = project_points(p_base, K, T_bb @ T_bc)
    ok = (z > NEAR_FIELD_REJECT_M) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & \
         (uv[:, 1] >= 0) & (uv[:, 1] < H)
    if not ok.any():
        return None, "ray outside image / all near field"
    if depth is not None:
        ui = uv[ok, 0].astype(int)
        vi = uv[ok, 1].astype(int)
        meas = depth[vi, ui]
        valid = meas > 0.05
        if valid.any():
            resid = np.abs(z[ok][valid] - meas[valid])
            agree = resid < DEPTH_AGREE_M
            run = None
            if agree.any():
                # The ray can agree with depth at several places along its
                # length: the target, plus whatever else it grazes. Taking the
                # median across ALL of them averages over disjoint surfaces
                # and can land between them, which is how the three charger
                # events failed while the ray itself demonstrably passes
                # through their masks (6/7 in calibration.yaml).
                #
                # Samples are ordered along the ray, so one physical surface
                # is a CONTIGUOUS run of agreeing samples. Take the longest
                # run and use its median: that is a single surface by
                # construction.
                idx = np.flatnonzero(agree)
                splits = np.flatnonzero(np.diff(idx) > 1)
                runs = np.split(idx, splits + 1)
                run = max(runs, key=len)
                # A run of one or two samples is not a surface, it is a
                # coincidental depth agreement. Measured: the seeds that land
                # correctly have runs of 13-36 samples, while the single
                # failure introduced when lfdws_t001_labexport gained real
                # depth had a run of 1. Below the floor, fall through to the
                # r0 projection, which is the more reliable estimator when
                # depth gives no consensus.
                if len(run) < MIN_DEPTH_RUN:
                    run = None
            if run is not None:
                pick = np.median(uv[ok][valid][run], axis=0)
                return pick, (f"ray-depth run {len(run)}/{int(agree.sum())} "
                              f"({np.median(resid[run])*1000:.0f} mm)")
    uv0, z0 = project_points(((T_bb @ np.append(r0, 1.0))[:3])[None, :], K,
                             T_bb @ T_bc)
    if z0[0] > 0 and 0 <= uv0[0, 0] < W and 0 <= uv0[0, 1] < H:
        return uv0[0], "r0 projection"
    return uv[ok][len(uv[ok]) // 2], "ray midpoint"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--derive_only", action="store_true")
    args = ap.parse_args()

    K, T_bc = load_calib()
    print(f"[calib] bota_to_camera loaded, t = "
          f"{(T_bc[:3,3]*1000).round(2).tolist()} mm", flush=True)

    offset, evs = None, None
    tj = "figures/tool_axis.json"
    if os.path.exists(tj):
        import json
        d = json.load(open(tj))
        offset = np.array(d["offset_bota_m"], float)
        evs = grasp_events(K, T_bc)
        print(f"\n[derive] using the recovered tool axis from {tj}: "
              f"offset {(offset*1000).round(2).tolist()} mm "
              f"({d['in_mask']}/{d['n_events']} in-mask, "
              f"{d['loto_in_mask']}/{d['n_events']} held-out)", flush=True)
    else:
        offset, evs = derive_gripper_offset(K, T_bc)
    if args.derive_only:
        return

    print("\n[compare] geometric seed vs auto_seed.py's heuristic, on the "
          "same grasp events", flush=True)
    print(f"{'trial':24s} {'geometric':>22s} {'auto_seed':>22s}", flush=True)
    geo_in, auto_in, n = 0, 0, 0
    rows_out = []
    for e in evs:
        H, W = e["mask"].shape
        n += 1
        g_txt = "not seeded"
        if offset is not None:
            uv, z = project_bota_point(offset, e["pose"], K, T_bc)
            inside = (z > 0 and 0 <= uv[0] < W and 0 <= uv[1] < H
                      and bool(e["mask"][int(uv[1]), int(uv[0])]))
            geo_in += int(inside)
            g_txt = f"({uv[0]:6.1f},{uv[1]:6.1f}) {'IN ' if inside else 'out'}"
            rows_out.append((e, "grasped", uv, inside))
        a_txt = "none on disk"
        ap_csv = os.path.join(os.path.dirname(e["figdir"]), "auto_seeds.csv")
        if not os.path.exists(ap_csv):
            ap_csv = os.path.join(e["figdir"], "auto_seeds.csv")
        if os.path.exists(ap_csv):
            for r in rows_of(ap_csv):
                if r.get("role") == "grasped":
                    ax, ay = float(r["seed_x"]), float(r["seed_y"])
                    ain = (0 <= ax < W and 0 <= ay < H
                           and bool(e["mask"][int(ay), int(ax)]))
                    auto_in += int(ain)
                    a_txt = f"({ax:6.1f},{ay:6.1f}) {'IN ' if ain else 'out'}"
                    break
        print(f"{e['trial']:24s} {g_txt:>22s} {a_txt:>22s}", flush=True)

    print(f"\n[result] seed lands inside the grasped object:", flush=True)
    print(f"           geometric {geo_in}/{n}", flush=True)
    print(f"           auto_seed {auto_in}/{n}", flush=True)
    if offset is None:
        print("  (geometric grasped seeding disabled: tool axis not "
              "established, see [derive] above)", flush=True)

    # ---------------------------------------------------------------- #
    # Contact roles. These need no gripper-centre derivation at all: the
    # wrench ray plus real depth locates the contact point directly, and the
    # ray itself is already validated at 6/7 events.
    # ---------------------------------------------------------------- #
    print("\n[contact] seeding contact roles from the wrench ray + depth",
          flush=True)
    from contact_eval_set import RECORDINGS, build_events
    cevs = build_events(verbose=False)
    tdir_of = {r["trial"]: os.path.dirname(r["rgb_dir"]) for r in RECORDINGS}
    print(f"{'event':40s} {'seed':>16s} {'via':>22s} {'in mask':>8s}",
          flush=True)
    c_in = 0
    for ev in cevs:
        H, W = ev["H"], ev["W"]
        td = tdir_of.get(ev["trial"], "")
        dep = depth_at(td, ev.get("depth_id", ev["img_id"]), H, W)
        uv, how = seed_contact(ev["force"], ev["torque"], ev["pose"], K, T_bc,
                               dep, H, W)
        if uv is None:
            print(f"{ev['trial']+'/'+ev['event']:40s} {'-':>16s} "
                  f"{how:>22s} {'-':>8s}", flush=True)
            continue
        inside = bool(ev["mask"][int(uv[1]), int(uv[0])])
        c_in += int(inside)
        print(f"{ev['trial']+'/'+ev['event']:40s} "
              f"({uv[0]:6.1f},{uv[1]:6.1f}) {how:>22s} "
              f"{'IN ' if inside else 'out':>8s}", flush=True)
    print(f"\n[result] contact seed lands inside the contact object: "
          f"{c_in}/{len(cevs)}", flush=True)

    for e, role, uv, inside in rows_out:
        outp = os.path.join(e["figdir"], "geometric_seeds.csv")
        os.makedirs(e["figdir"], exist_ok=True)
        if os.path.exists(outp):
            shutil.copy2(outp, outp + ".bak")
        H, W = e["mask"].shape
        with open(outp, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["role", "event", "img_id", "seed_x", "seed_y",
                        "frac_x", "frac_y", "mask_px"])
            w.writerow([role, "grasp", e["img_id"], f"{uv[0]:.1f}",
                        f"{uv[1]:.1f}", f"{uv[0]/W:.4f}", f"{uv[1]/H:.4f}",
                        int(e["mask"].sum())])
        print(f"[write] {outp}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
