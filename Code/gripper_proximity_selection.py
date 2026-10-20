"""3D gripper-proximity selection, adapted from SPARC to an eye-in-hand rig.

WHY
---
SPARC (arXiv:2606.13497) selects the manipulated object with
    R_i = w_A*A_i + w_D(P_i)*D_i + w_P*P_i - Pi_i
where P_i, "gripper proximity", is the fraction of a candidate's visible 3D
points falling inside a sphere centred on the gripper.

Its primary cue, phase-aware motion A_i = (mu_int^0.6)/((mu_non+1)^0.2), is
object motion during the interaction divided by motion outside it, and SPARC
states plainly that it "assumes a mostly static camera". On a wrist-mounted
camera everything moves in every frame, mu_non is large on every candidate,
and A_i collapses. That is the same failure already measured here for
DistinctNet (0/5) and for our own image-stationarity attempts
(archive/negative_results/motion_selection_*.py, which lose to random).

P_i does NOT depend on a static camera. It is the cue worth porting.

WHY IT SHOULD BEAT WHAT IS ALREADY TRIED
-----------------------------------------
The frozen selector's spatial cue is IoU between a proposal and a hand-drawn
2D polygon, which is scale-biased: measured here, it wins on the large cube
(3/3) and scores 0/18 on the small nuts, because a small object fully inside a
large region still has tiny IoU with it. Its complement, containment, fails the
opposite way (0/5): every tiny fragment inside the region scores 1.0.

P_i is a FRACTION OF THE CANDIDATE'S OWN POINTS, in 3D, so object size cancels.
A small nut entirely beside the gripper scores near 1.0; a large background
region with a few near pixels scores low.

PRE-REGISTERED, FIXED BEFORE THE FIRST RUN (2026-10-20)
---------------------------------------------------------
Written down before execution because every earlier selector here failed by
being tuned against the same five answers.

  GRIPPER POINT g
      On an eye-in-hand rig the gripper is the closest thing to the camera, so
      g is the 3D centroid of the valid depth points at or below the
      NEAREST_PERCENTILE-th depth percentile. No extrinsic, no robot
      segmentation, no hand-drawn region. SPARC uses a RobotSeg mask for this;
      we have none, and the eye-in-hand geometry supplies the substitute.
  RADIUS r = 0.15 m
      Physically motivated, not fitted: the Franka Hand opens to 0.08 m, so a
      held object lies within ~0.04 m of the gripper axis, plus margin for
      depth noise and object extent. SPARC gives no closed form for its own
      radius.
  SCORE
      P_i = (candidate's valid 3D points within r of g) / (its valid points).
      Candidates with fewer than MIN_POINTS valid points are abstained on
      rather than scored, since a fraction over a handful of pixels is noise.
  DECISION
      argmax P_i. Correct at IoU >= 0.50 against the reference mask.
  MARKS
      3/3 of the depth-bearing grasped cases -> port it into the selector.
      2/3 -> matches the best hand-written rule (2/5 overall); inconclusive.
      <=1/3 -> the cue does not transfer; say so and stop.

The radius sweep this script also prints is a SENSITIVITY DIAGNOSTIC. The
result is the r = 0.15 m row. Reporting the best row of the sweep would be
exactly the fitting this protocol exists to prevent.

SCOPE
-----
Only three of the five grasped cases have usable metric depth: lfdws_t002_new,
lfdws_t004, lfdws_t005 (three independent groups). lfdws_t001 has no depth
stream, and lfdws_t002_labexport's "depth" is an 8-bit 3-255 visualisation,
not metric. Those two are skipped, not scored as failures.

Usage:
    .venv_analysis/bin/python Code/gripper_proximity_selection.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from event_utils import mask_from_overlay

BUNDLE = "figures/interaction_bakeoff/input"
MANIFEST = "config/evaluation_manifest.yaml"
CALIB = "calibration.yaml"
OUT = "figures/gripper_proximity_result.txt"

IMG_COL = "zed.zed_node.rgb.color.rect.image.compressed"
DEPTH_COL = "zed.zed_node.depth.depth_registered.compressedDepth"
GRASPED_BGR = (0, 255, 0)

# ---- pre-registered constants, fixed 2026-10-20 before any run -------------
RADIUS_M = 0.15
NEAREST_PERCENTILE = 2.0
MIN_POINTS = 50
POSITIVE_IOU = 0.50
SWEEP = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]   # diagnostic only

TRIALS = {
    "lfdws_t001": "Data/lfdws_t001/lfdws_t001",
    "lfdws_t001_depth": "Data/lfdws_t001_depth",
    "lfdws_t001_labexport": "Data/lfdws_t001_labexport/lfdws_t001",
    "lfdws_t002_new": "Data/lfdws_t002_new",
    "lfdws_t002_labexport": "Data/lfdws_t002_labexport/lfdws_t002",
    "lfdws_t004": "Data/lfdws_t004",
    "lfdws_t005": "Data/lfdws_t005",
}


def log(m):
    print(m, flush=True)


def iou(a, b):
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def depth_filename_for(trial, img_id):
    """Map an RGB frame id to its depth frame id via the merged CSV."""
    cs = glob.glob(os.path.join(trial, "*_0.csv"))
    if not cs:
        return None
    with open(cs[0]) as fh:
        for row in csv.DictReader(fh):
            if str(row.get(IMG_COL, "")).strip() == img_id:
                v = str(row.get(DEPTH_COL, "")).strip()
                return v or None
    return None


def load_depth_m(trial, depth_id, shape):
    """Return metric depth in metres at `shape`, or None if not usable."""
    hits = glob.glob(os.path.join(trial, "**", "*depth*", depth_id + ".png"),
                     recursive=True)
    if not hits:
        return None
    d = cv2.imread(hits[0], cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    if d.dtype != np.uint16:
        # 8-bit depth is a visualisation, not metric. Refuse it rather than
        # silently producing nonsense geometry.
        return None
    if d.shape[:2] != shape:
        d = cv2.resize(d, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    m = d.astype(np.float32) / 1000.0
    m[d == 0] = np.nan
    return m


def backproject(depth_m, K):
    h, w = depth_m.shape
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    vs, us = np.mgrid[0:h, 0:w]
    Z = depth_m
    X = (us - cx) * Z / fx
    Y = (vs - cy) * Z / fy
    return np.dstack([X, Y, Z])


def main():
    calib = yaml.safe_load(open(CALIB))
    K = calib["camera_intrinsics"]["K"]
    man = yaml.safe_load(open(MANIFEST))
    refs = {r["id"]: r for r in man["recordings"]}
    bench = json.load(open(os.path.join(BUNDLE, "benchmark.json")))
    cases = [c for c in bench["cases"] if c["role"] == "grasped"]

    lines = []

    def out(s):
        log(s)
        lines.append(s)

    out("3D GRIPPER-PROXIMITY SELECTION (SPARC cue, eye-in-hand adaptation)")
    out(f"pre-registered: r={RADIUS_M} m, g=centroid of nearest "
        f"{NEAREST_PERCENTILE}% depth, min {MIN_POINTS} pts, IoU>={POSITIVE_IOU}")
    out("")

    rows, sweep_hits = [], {r: 0 for r in SWEEP}
    n_ok = n_scored = 0

    for case in cases:
        cid, rid, img_id = case["case_id"], case["recording_id"], case["img_id"]
        trial = TRIALS[rid]
        img = cv2.imread(os.path.join(BUNDLE, case["image"]))
        if img is None:
            out(f"{cid:<40} SKIP unreadable image")
            continue
        shape = img.shape[:2]

        did = depth_filename_for(trial, img_id)
        depth = load_depth_m(trial, did, shape) if did else None
        if depth is None:
            out(f"{cid:<40} SKIP no usable metric depth")
            continue

        rec = refs.get(rid, {})
        ref = None
        summary = rec.get("reference_sidecar")
        if summary and os.path.exists(summary):
            with open(summary) as fh:
                for r in csv.DictReader(fh):
                    if r.get("role") == "grasped" and \
                       os.path.splitext(r["img_filename"])[0] == img_id:
                        ov = r.get("overlay_path")
                        src = os.path.join(
                            trial, "zed_zed_node_rgb_color_rect_image_compressed",
                            img_id + ".png")
                        if ov and os.path.exists(ov) and os.path.exists(src):
                            ref = mask_from_overlay(ov, src, GRASPED_BGR)
                        break
        if ref is None or not ref.any():
            out(f"{cid:<40} SKIP no reference mask")
            continue

        pts = backproject(depth, K)
        valid = np.isfinite(depth)
        # gripper point: nearest depth percentile, 3D centroid
        thr = np.nanpercentile(depth[valid], NEAREST_PERCENTILE)
        near = valid & (depth <= thr)
        g = np.array([np.nanmean(pts[..., k][near]) for k in range(3)])

        masks = np.load(os.path.join(BUNDLE, case["proposal_cache"]),
                        allow_pickle=True)["masks"]
        dist_all = np.linalg.norm(pts - g, axis=2)

        def score_at(radius):
            best, best_i = -1.0, None
            for i in range(masks.shape[0]):
                m = masks[i].astype(bool) & valid
                n = int(m.sum())
                if n < MIN_POINTS:
                    continue
                p = float((dist_all[m] <= radius).sum()) / n
                if p > best:
                    best, best_i = p, i
            return best, best_i

        p_star, idx = score_at(RADIUS_M)
        if idx is None:
            out(f"{cid:<40} ABSTAIN no candidate with >= {MIN_POINTS} depth pts")
            continue
        v = iou(masks[idx].astype(bool), ref)
        ok = v >= POSITIVE_IOU
        n_scored += 1
        n_ok += ok
        rows.append((cid, case["independent_group"], v, p_star, ok))
        out(f"{cid:<40} g=({g[0]:+.2f},{g[1]:+.2f},{g[2]:.2f})m  "
            f"P={p_star:.3f}  IoU={v:.3f}  {'CORRECT' if ok else 'wrong'}")

        for r in SWEEP:
            _, i2 = score_at(r)
            if i2 is not None and iou(masks[i2].astype(bool), ref) >= POSITIVE_IOU:
                sweep_hits[r] += 1

    out("")
    out("=" * 72)
    out(f"RESULT (pre-registered r={RADIUS_M} m): {n_ok}/{n_scored} correct")
    out("")
    out("Comparison on the same grasped role:")
    out("  region-IoU alone (all 5 cases)              2/5")
    out("  region-containment alone                    0/5")
    out("  17-feature fitted, group-held-out           1/5")
    out("  Molmo 2 -> SAM 2, pre-registered rule       1/5")
    out(f"  gripper proximity (this, depth cases only)  {n_ok}/{n_scored}")
    out("")
    out("SENSITIVITY DIAGNOSTIC - not the result, not evidence:")
    for r in SWEEP:
        mark = "   <- pre-registered" if abs(r - RADIUS_M) < 1e-9 else ""
        out(f"  r={r:<5} m  {sweep_hits[r]}/{n_scored}{mark}")
    out("")
    if n_scored == 0:
        verdict = "NO CASES SCORED - depth mapping or references unavailable."
    elif n_ok == n_scored and n_scored >= 3:
        verdict = (f"PASS {n_ok}/{n_scored} - port the cue into the selector, "
                   "then re-measure on all five cases.")
    elif n_ok >= 2:
        verdict = (f"INCONCLUSIVE {n_ok}/{n_scored} - matches the best "
                   "hand-written rule, does not beat it.")
    else:
        verdict = (f"FAIL {n_ok}/{n_scored} - the cue does not transfer to "
                   "this rig. Record it and stop.")
    out(f"VERDICT: {verdict}")
    out("")
    out("Standing caveat: 3 depth-bearing cases across 3 independent groups. "
        "A pass says the cue is worth porting, not that it generalises.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
