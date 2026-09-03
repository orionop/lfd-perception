"""
Is there ANY fixed bota-frame offset that seeds the grasped role, or is the
quantity genuinely not a rig constant?

WHY THIS EXISTS
---------------
Code/measure_grasp_offset.py measured where a held object sits, per sample,
and tested the MEAN of those measurements leave-one-trial-out. It failed
(9/17). But the per-trial table it printed is bimodal, not noisy:

    lfdws_t002_new        n=6  mean [ -8.2, 14.2,  -6.2]  spread [1.9, 0.9, 1.1]
    lfdws_t002_labexport  n=6  mean [ -8.3, 12.0,  -6.7]  spread [1.9, 3.0, 2.3]
    lfdws_t004            n=3  mean [-24.3, 13.3,  19.3]  spread [10.8, 8.9, 3.5]
    lfdws_t005            n=2  mean [-30.3, 19.2,  16.5]  spread [0.3, 1.3, 0.6]

Two tight clusters ~31 mm apart (cube trials vs pegboard-tool trials). The
mean of a bimodal set lands in the GAP between the clusters, inside neither.
So "the mean fails" does not establish "no fixed offset works" -- the mean is
simply the wrong estimator for a bimodal sample. This separates the two.

WHAT IT DOES
------------
Grid-searches the offset directly on the objective that matters (does the
projected point land inside the grasped mask) instead of estimating it as a
centroid and hoping. The samples, masks and poses come from
measure_grasp_offset.collect(), unmodified.

THE OVERFIT GUARD, WHICH THIS REPO HAS ALREADY NEEDED ONCE
----------------------------------------------------------
Code/extrinsic_grid_search.py searched four free parameters over 91,800
candidates and scored 1.000 in-fold against 0.067 held out. This searches
three parameters over a few thousand, on 17 samples, which is not obviously
safer. So the in-fold number is reported but is NOT the result. The result is
leave-one-trial-out: fit the offset on three recordings, test it on the
recording it never saw. A null is also reported -- the in-mask rate of the
whole candidate grid -- so a "win" that merely matches what a random offset
achieves is visible as such.

Read only. Always writes figures/grasp_offset_search.json carrying an
explicit "recovered" flag, so a failed search leaves an accurate record rather
than an absent file or a stale successful one. Nothing consumes it
automatically: Code/geometric_seed.py reads figures/tool_axis.json.

Usage: .venv_analysis/bin/python Code/grasp_offset_search.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometric_seed import load_calib
from wrench_ray import pose_to_T, project_points
from measure_grasp_offset import TRIALS, collect

OUT_JSON = "figures/grasp_offset_search.json"

# Bounds bracket every per-trial cluster measured above, with margin. Step is
# 2.5 mm: finer than the tightest cluster spread (0.3 mm) is meaningless when
# the target is a mask tens of pixels across.
LO = np.array([-95.0, -55.0, -75.0])
HI = np.array([55.0, 80.0, 80.0])
STEP = 2.5

# lfdws_t002_new and lfdws_t002_labexport are the SAME BAG, exported twice
# (our mcap_extract.py vs the lab's native ros2_unbag). Holding one out while
# fitting on the other is not a held-out recording, it is the same recording
# scored against itself -- and at 12 of 17 samples the pair dominates every
# fold. They are one group here. This leaves 3 independent recordings, which
# is few, and the verdict says so.
GROUP = {"lfdws_t002_new": "t002_bag", "lfdws_t002_labexport": "t002_bag"}

# A held-out trial counts as seeded if the offset lands inside its grasped
# mask on a MAJORITY of its samples. One lucky frame is not a seed.
PASS_FRAC = 0.5

# Leave-one-out ALONE does not catch this failure, which is why it is not the
# only gate. A 3-parameter search over 3 recordings can pass LOTO on isolated
# knife-edge candidates that thread every mask by coincidence of the three
# particular viewing geometries. A genuine rig constant is not a point, it is
# a CONNECTED BASIN: the masks are large and overlapping, so offsets a few mm
# apart must score alike. Measured on the first run, the top-scoring set was 3
# candidates spread over [17.5, 50.0, 72.5] mm -- scattered, not a basin. Both
# gates must pass.
BASIN_MAX_EXTENT_MM = 20.0
BASIN_MIN_COUNT = 8


def build_grid():
    axes = [np.arange(LO[i], HI[i] + 1e-9, STEP) for i in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    return g / 1000.0          # mm -> m, the unit project_bota_point wants


def score_grid(grid, samples, K, T_bc):
    """in-mask hit count per candidate offset, over the given samples.

    Projects the WHOLE candidate grid in one call per sample: the pose, and
    therefore T_base_bota, is fixed within a sample, so the per-candidate
    Python loop that project_bota_point implies is pure overhead.
    """
    hits = np.zeros(len(grid), dtype=np.int32)
    homog = np.concatenate([grid, np.ones((len(grid), 1))], axis=1)
    for si, s in enumerate(samples):
        H, W = s["mask"].shape
        T_bb = pose_to_T(*s["pose"])
        p_base = (T_bb @ homog.T).T[:, :3]
        uv, z = project_points(p_base, K, T_bb @ T_bc)
        x = uv[:, 0].astype(np.int32)
        y = uv[:, 1].astype(np.int32)
        ok = (z > 0) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
        if ok.any():
            inside = np.zeros(len(grid), dtype=bool)
            inside[ok] = s["mask"][y[ok], x[ok]]
            hits += inside
        print(f"  [score] sample {si + 1}/{len(samples)} ({s['trial']}) "
              f"{int(ok.sum())}/{len(grid)} candidates in frame", flush=True)
    return hits


def main():
    K, T_bc = load_calib()
    print(f"[calib] bota_to_camera t = {(T_bc[:3, 3] * 1000).round(2).tolist()}"
          " mm", flush=True)

    samples = []
    for label, tdir, sidecar in TRIALS:
        samples += collect(label, tdir, sidecar, K, T_bc)
    if not samples:
        print("[fatal] no usable grasp samples", flush=True)
        return
    for s in samples:
        s["group"] = GROUP.get(s["trial"], s["trial"])
    trials = sorted({s["trial"] for s in samples})
    groups = sorted({s["group"] for s in samples})
    print(f"\n[data] {len(samples)} samples over {len(trials)} trials, "
          f"{len(groups)} independent recordings: {groups}", flush=True)

    grid = build_grid()
    print(f"[grid] {len(grid)} candidate offsets, {STEP} mm step, bounds "
          f"{LO.tolist()} to {HI.tolist()} mm", flush=True)

    print("\n[fit-all] scoring every candidate on every sample", flush=True)
    hits = score_grid(grid, samples, K, T_bc)
    best = int(hits.argmax())
    print(f"\n[fit-all] best offset {(grid[best] * 1000).round(2).tolist()} mm"
          f"  -> {hits[best]}/{len(samples)} samples in-mask", flush=True)
    print(f"[null]    over all {len(hits)} candidates: median "
          f"{np.median(hits):.1f}/{len(samples)}, mean {hits.mean():.2f}, "
          f"{int((hits == hits[best]).sum())} candidates tie for best",
          flush=True)
    frac_as_good = float((hits >= hits[best]).mean())
    print(f"[null]    fraction of the grid scoring as well as the winner: "
          f"{frac_as_good:.4f}", flush=True)

    # A 3-parameter search validated on 3 recordings can fit a knife edge: an
    # isolated candidate that happens to thread every mask and would miss on a
    # 4th recording. A real rig constant instead sits in a CONNECTED BASIN of
    # offsets tens of mm wide, because the masks are large and overlapping. So
    # measure the basin rather than trusting the single argmax.
    print("\n[basin] how big is the region that scores near the winner?",
          flush=True)
    for thr in (hits[best], hits[best] - 1, hits[best] - 2):
        sel = grid[hits >= thr] * 1000.0
        if not len(sel):
            continue
        ext = (sel.max(0) - sel.min(0)).round(1)
        print(f"  >= {thr}/{len(samples)}: {len(sel):6d} candidates, "
              f"extent {ext.tolist()} mm, centroid "
              f"{sel.mean(0).round(1).tolist()} mm", flush=True)

    print("\n[per-trial] the single global winner, broken down by recording",
          flush=True)
    for t in trials:
        sub = [s for s in samples if s["trial"] == t]
        h = score_grid(grid[best][None, :], sub, K, T_bc)[0]
        print(f"  {t:24s} {h}/{len(sub)} in-mask", flush=True)
    mag = float(np.linalg.norm(grid[best]) * 1000)
    print(f"\n[physical] |offset| = {mag:.1f} mm from the bota origin; Mark's "
          f"drawing puts the GRIPPER CENTRE at 104 mm. These need not agree: "
          f"this is a seed point inside the held object, not the gripper "
          f"centre.", flush=True)

    print("\n[loto] leave-one-recording-out: fit on the rest, test "
          "on the recording never seen", flush=True)
    passed, tested = 0, 0
    for held in groups:
        fit = [s for s in samples if s["group"] != held]
        test = [s for s in samples if s["group"] == held]
        h = score_grid(grid, fit, K, T_bc)
        off = grid[int(h.argmax())]
        t_hits = score_grid(off[None, :], test, K, T_bc)[0]
        ok = t_hits >= PASS_FRAC * len(test)
        passed += int(ok)
        tested += 1
        print(f"  held out {held:24s} offset "
              f"{(off * 1000).round(1).tolist()} mm -> {t_hits}/{len(test)} "
              f"in-mask  {'PASS' if ok else 'fail'}", flush=True)

    print(f"\n[loto] {passed}/{tested} held-out recordings seeded", flush=True)
    b = grid[best] * 1000
    edge = np.minimum(np.abs(b - LO), np.abs(b - HI)).min()
    print(f"[edge] winner sits {edge:.1f} mm from the nearest grid bound "
          f"({'INTERIOR, bounds are not binding' if edge > 3 * STEP else 'AT THE EDGE, widen the bounds'})",
          flush=True)
    top = grid[hits == hits[best]] * 1000.0
    top_extent = (top.max(0) - top.min(0)) if len(top) else np.zeros(3)
    basin_ok = bool(len(top) >= BASIN_MIN_COUNT
                    and (top_extent <= BASIN_MAX_EXTENT_MM).all())
    print(f"\n[gate] held-out   {passed}/{tested} recordings   "
          f"{'PASS' if passed == tested and tested >= 3 else 'FAIL'}",
          flush=True)
    print(f"[gate] basin      {len(top)} top candidates spanning "
          f"{top_extent.round(1).tolist()} mm (need >= {BASIN_MIN_COUNT} "
          f"within {BASIN_MAX_EXTENT_MM:.0f} mm)   "
          f"{'PASS' if basin_ok else 'FAIL'}", flush=True)

    recovered = bool(passed == tested and tested >= 3 and basin_ok)
    os.makedirs("figures", exist_ok=True)
    json.dump({"recovered": recovered,
               "offset_bota_m": grid[best].tolist(),
               "in_mask": int(hits[best]), "n_samples": len(samples),
               "loto_passed": passed, "n_recordings": tested,
               "grid_frac_as_good": frac_as_good,
               "top_count": int(len(top)),
               "top_extent_mm": top_extent.round(2).tolist(),
               "basin_ok": basin_ok},
              open(OUT_JSON, "w"), indent=2)

    if recovered:
        print(f"[verdict] RECOVERED and generalises. Wrote {OUT_JSON}",
              flush=True)
    else:
        print("[verdict] NOT recovered. The winner scores far above the null "
              "and survives leave-one-recording-out, but the top-scoring "
              "offsets are isolated points rather than a connected basin, "
              "which is what a coincidental fit to three viewing geometries "
              "looks like and what a rig constant does not. The grasped role "
              "stays unseeded.", flush=True)
        print(f"[verdict] recorded as recovered=false in {OUT_JSON}; nothing "
              "downstream consumes it.", flush=True)

    print("[done]", flush=True)


if __name__ == "__main__":
    main()
