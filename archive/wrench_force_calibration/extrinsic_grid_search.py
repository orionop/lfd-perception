"""
Recover T_bota_camera by searching the small space Mark's CAD left open.

WHY THIS IS NEWLY POSSIBLE
--------------------------
Until 2026-08-18 the bota-to-camera transform was wholly unconstrained, and
two CAD-derived candidates both projected the wrench ray entirely outside the
image on every trial (Code/cad_candidate_sensitivity.py). Mark's rev2
dimensions fix the gripper-centre-to-camera translation at
(0, 158.82, 91.72) mm, verified internally consistent to 8 microns. What is
left free is 3 bounded continuous parameters plus a discrete lens choice, all
described in Code/wrench_ray.py. That is small enough to enumerate.

THE HONESTY PROBLEM, AND WHAT IS DONE ABOUT IT
----------------------------------------------
Fitting an extrinsic to the same masks used to judge it is circular. An
earlier version of the calibration sensitivity analysis was thrown out for
exactly this. Three guards, all mandatory, none optional:

  1. r0 . fhat == 0 is asserted before any search runs. Bicchi's r0 is
     orthogonal to the force direction by construction, so if that fails the
     geometry code is wrong and no search result would mean anything.

  2. LEAVE-ONE-RECORDING-OUT cross-validation. The optimum is chosen on two
     recordings and scored on the third, which it never saw. The held-out
     number is the one reported. A transform that only works in-fold is a
     failure and is labelled as one.

  3. AN EMPIRICAL NULL. The full grid is itself the null distribution: if the
     median candidate already hits 60 percent of events, a best candidate at
     100 percent proves nothing. The best score is reported as a percentile of
     the grid, and the landscape is checked for whether the optimum is a
     single sharp peak or a broad plateau. A plateau means the data does not
     determine the transform and we wait for Mark's angle.

With 7 events across 3 recordings this can support "the data is consistent
with X" and cannot support "the extrinsic is X". Independent confirmation
still requires Mark's CAD angle or a ChArUco hand-eye calibration.

DOES NOT WRITE calibration.yaml. Prints for review, same rule
Code/calibrate_hand_eye.py follows.

Usage:
    .venv_analysis/bin/python Code/extrinsic_grid_search.py
    .venv_analysis/bin/python Code/extrinsic_grid_search.py --psi_step 5 --tilt_step 1
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_eval_set import build_events
from wrench_ray import (D_AXIAL_MAX_MM, D_AXIAL_MIN_MM,
                        LENS_SHIFT_CANDIDATES_MM, make_T_bota_camera,
                        pose_to_T, wrench_line_bota)

OUT_CSV = "figures/extrinsic_grid_search.csv"
OUT_PNG = "figures/extrinsic_grid_search.png"
CALIB = "calibration.yaml"

# Ray sampling. Both signs of s are covered because the sensor measures the
# reaction wrench and the contact may lie either way along the line of action.
S_MIN, S_MAX, S_N = -0.60, 0.60, 120

# Candidates evaluated per chunk, to bound peak memory.
CHUNK = 8000


def load_K(path=CALIB):
    import yaml
    with open(path) as f:
        c = yaml.safe_load(f)
    intr = c["camera_intrinsics"]
    if not intr.get("filled", False):
        print("[fatal] camera_intrinsics is not filled:true", flush=True)
        sys.exit(1)
    return np.array(intr["K"], dtype=float)


def self_test():
    """Bicchi's r0 must be orthogonal to fhat, exactly, at zero noise."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(500):
        f = rng.normal(size=3) * 10.0
        contact = rng.normal(size=3) * 0.2
        tau = np.cross(contact, f)          # pure force applied at `contact`
        r0, fhat = wrench_line_bota(f, tau)
        worst = max(worst, abs(float(r0 @ fhat)))
        # the recovered line must also pass through the true contact point
        resid = np.linalg.norm(np.cross(contact - r0, fhat))
        worst = max(worst, resid)
    print(f"[selftest] max |r0.fhat| and line residual = {worst:.3e}",
          flush=True)
    if worst > 1e-9:
        print("[fatal] Bicchi recovery is not exact at zero noise", flush=True)
        sys.exit(1)


def build_grid(args):
    psi = np.arange(0.0, 360.0, args.psi_step)
    tilt = np.arange(args.tilt_min, args.tilt_max + 1e-9, args.tilt_step)
    dax = np.arange(D_AXIAL_MIN_MM, D_AXIAL_MAX_MM + 1e-9, args.d_step)
    lens = np.array(LENS_SHIFT_CANDIDATES_MM, dtype=float)
    P, T, D, L = np.meshgrid(psi, tilt, dax, lens, indexing="ij")
    return np.stack([P.ravel(), T.ravel(), D.ravel(), L.ravel()], axis=1)


def transforms_for(grid):
    """(M,4,4) candidate T_bota_camera for every grid row."""
    M = len(grid)
    Ts = np.empty((M, 4, 4), dtype=float)
    for i in range(M):
        Ts[i] = make_T_bota_camera(grid[i, 0], grid[i, 1],
                                   grid[i, 2], grid[i, 3])
        if (i + 1) % 20000 == 0:
            print(f"  [build] {i+1}/{M} transforms", flush=True)
    return Ts


def score_event(ev, Ts, K):
    """(hit, dist) over all candidates for one event, vectorised.

    T_bota_camera does not depend on the event, so the ray is sampled once in
    the bota frame and pushed through every candidate at once.
    """
    r0, fhat = wrench_line_bota(ev["force"], ev["torque"])
    s = np.linspace(S_MIN, S_MAX, S_N)
    pts_bota = r0[None, :] + s[:, None] * fhat[None, :]        # (N,3)

    p = ev["pose"]
    T_base_bota = pose_to_T(*p)
    R_bb, t_bb = T_base_bota[:3, :3], T_base_bota[:3, 3]
    pts_base = pts_bota @ R_bb.T + t_bb                        # (N,3)

    mask, H, W = ev["mask"], ev["H"], ev["W"]
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    fx, fy, ppx, ppy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    M = len(Ts)
    hit = np.zeros(M, dtype=bool)
    dist = np.full(M, np.inf, dtype=float)

    for a in range(0, M, CHUNK):
        b = min(a + CHUNK, M)
        Tc = Ts[a:b]                                           # (m,4,4)
        R_bc, t_bc = Tc[:, :3, :3], Tc[:, :3, 3]
        # T_base_cam = T_base_bota @ T_bota_cam ; invert to map base -> cam
        R_basecam = R_bb @ R_bc                                # (m,3,3)
        t_basecam = (R_bb @ t_bc.T).T + t_bb                   # (m,3)
        d = pts_base[None, :, :] - t_basecam[:, None, :]       # (m,N,3)
        cam = np.einsum("mij,mnj->mni", np.transpose(R_basecam, (0, 2, 1)), d)

        z = cam[:, :, 2]
        front = z > 1e-3
        zs = np.where(front, z, 1.0)
        u = fx * cam[:, :, 0] / zs + ppx
        v = fy * cam[:, :, 1] / zs + ppy
        inb = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)

        ui = np.clip(u, 0, W - 1).astype(np.int32)
        vi = np.clip(v, 0, H - 1).astype(np.int32)
        inside = mask[vi, ui] & inb
        hit[a:b] = inside.any(axis=1)

        dd = np.where(inb, np.hypot(u - cx, v - cy), np.inf)
        dist[a:b] = dd.min(axis=1)

    return hit, dist


def summarise(name, sel, HITS, DIST, events):
    """Best candidate over the event subset `sel`, plus the empirical null."""
    h = HITS[:, sel]
    d = DIST[:, sel]
    rate = h.mean(axis=1)
    md = np.where(np.isfinite(d), d, 1e6).mean(axis=1)
    order = np.lexsort((md, -rate))
    best = order[0]
    pct = float((rate <= rate[best]).mean() * 100.0)
    print(f"  [{name}] best hit rate {rate[best]:.3f} "
          f"({int(h[best].sum())}/{len(sel)} events), "
          f"mean centroid dist {md[best]:8.1f} px", flush=True)
    print(f"  [{name}] grid null: median hit rate {np.median(rate):.3f}, "
          f"90th pct {np.percentile(rate, 90):.3f}, "
          f"best sits at the {pct:.2f}th percentile", flush=True)
    return best, rate, md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--psi_step", type=float, default=10.0)
    ap.add_argument("--tilt_step", type=float, default=3.0)
    ap.add_argument("--tilt_min", type=float, default=-30.0)
    ap.add_argument("--tilt_max", type=float, default=120.0)
    ap.add_argument("--d_step", type=float, default=4.0)
    args = ap.parse_args()

    print("[stage] self-test on the Bicchi recovery", flush=True)
    self_test()

    print("\n[stage] building the shared contact event set", flush=True)
    events = build_events()
    if len(events) < 4:
        print(f"[fatal] only {len(events)} usable events, not enough to "
              f"cross-validate", flush=True)
        sys.exit(1)
    trials = sorted(set(e["trial"] for e in events))
    print(f"[events] {len(events)} events across {len(trials)} recordings: "
          f"{trials}", flush=True)
    chance = np.mean([e["mask_frac"] for e in events])
    print(f"[events] mean mask coverage {chance:.3%} of frame "
          f"(a ray placed at random is unlikely to hit by accident)",
          flush=True)

    K = load_K()
    grid = build_grid(args)
    print(f"\n[stage] grid: {len(grid):,} candidates "
          f"(psi step {args.psi_step}, tilt step {args.tilt_step}, "
          f"d step {args.d_step}, {len(LENS_SHIFT_CANDIDATES_MM)} lens "
          f"offsets)", flush=True)
    Ts = transforms_for(grid)

    HITS = np.zeros((len(grid), len(events)), dtype=bool)
    DIST = np.zeros((len(grid), len(events)), dtype=float)
    for j, ev in enumerate(events):
        print(f"[score] {j+1}/{len(events)} {ev['trial']}/{ev['event']} ...",
              flush=True)
        h, d = score_event(ev, Ts, K)
        HITS[:, j], DIST[:, j] = h, d
        print(f"  [score] {int(h.sum()):,}/{len(grid):,} candidates hit this "
              f"event ({h.mean():.2%} of the grid)", flush=True)

    print("\n[stage] fit on all events (IN-FOLD, not evidence)", flush=True)
    all_idx = np.arange(len(events))
    best, rate, md = summarise("all", all_idx, HITS, DIST, events)
    g = grid[best]
    print(f"  [all] psi={g[0]:.1f} deg  tilt={g[1]:.1f} deg  "
          f"d_axial={g[2]:.1f} mm  lens_shift={g[3]:+.1f} mm", flush=True)

    print("\n[stage] leave-one-recording-out cross-validation", flush=True)
    loto = []
    for held in trials:
        tr = np.array([i for i, e in enumerate(events) if e["trial"] != held])
        te = np.array([i for i, e in enumerate(events) if e["trial"] == held])
        if len(tr) == 0 or len(te) == 0:
            continue
        b, _, _ = summarise(f"fit-without-{held}", tr, HITS, DIST, events)
        held_rate = float(HITS[b, te].mean())
        gg = grid[b]
        print(f"  [held-out {held}] hit rate {held_rate:.3f} "
              f"({int(HITS[b, te].sum())}/{len(te)}) with "
              f"psi={gg[0]:.1f} tilt={gg[1]:.1f} d={gg[2]:.1f} "
              f"lens={gg[3]:+.1f}", flush=True)
        loto.append((held, held_rate, gg))

    print("\n[stage] is the optimum a peak or a plateau?", flush=True)
    top = np.flatnonzero(rate >= rate[best] - 1e-9)
    print(f"  {len(top):,} candidates tie at the best hit rate "
          f"({rate[best]:.3f})", flush=True)
    if len(top):
        for k, nm in enumerate(["psi", "tilt", "d_axial", "lens_shift"]):
            vals = grid[top, k]
            print(f"    {nm:11s} spread {vals.min():8.1f} .. {vals.max():8.1f}"
                  f"   unique={len(np.unique(vals))}", flush=True)

    os.makedirs("figures", exist_ok=True)
    if os.path.exists(OUT_CSV):
        import shutil
        shutil.copy2(OUT_CSV, OUT_CSV + ".bak")
        print(f"\n[backup] {OUT_CSV} -> {OUT_CSV}.bak", flush=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "psi_deg", "tilt_deg", "d_axial_mm",
                    "lens_shift_mm", "hit_rate", "mean_centroid_dist_px"])
        order = np.lexsort((md, -rate))[:200]
        for r, i in enumerate(order):
            w.writerow([r, f"{grid[i,0]:.2f}", f"{grid[i,1]:.2f}",
                        f"{grid[i,2]:.2f}", f"{grid[i,3]:.2f}",
                        f"{rate[i]:.4f}", f"{md[i]:.2f}"])
    print(f"[write] {OUT_CSV} (top 200 candidates)", flush=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        ax[0].hist(rate, bins=np.linspace(0, 1, 30), color="#888")
        ax[0].axvline(rate[best], color="#c02626", lw=2,
                      label=f"best {rate[best]:.3f}")
        ax[0].set_xlabel("hit rate over all events")
        ax[0].set_ylabel("candidates")
        ax[0].set_title("empirical null: the grid itself")
        ax[0].legend()
        # marginal of hit rate against tilt, the parameter that binds
        tl = np.unique(grid[:, 1])
        prof = [rate[grid[:, 1] == t].max() for t in tl]
        ax[1].plot(tl, prof, color="#2a7240")
        ax[1].set_xlabel("camera tilt below horizontal (deg)")
        ax[1].set_ylabel("best hit rate at this tilt")
        ax[1].set_title("peak or plateau?")
        fig.tight_layout()
        if os.path.exists(OUT_PNG):
            import shutil
            shutil.copy2(OUT_PNG, OUT_PNG + ".bak")
        fig.savefig(OUT_PNG, dpi=140)
        print(f"[write] {OUT_PNG}", flush=True)
    except Exception as e:
        print(f"[warn] figure skipped: {type(e).__name__}: {e}", flush=True)

    print("\n[verdict]", flush=True)
    if loto:
        mean_held = float(np.mean([r for _, r, _ in loto]))
        print(f"  mean held-out hit rate = {mean_held:.3f}", flush=True)
        if mean_held >= 0.8 and len(top) < 0.01 * len(grid):
            print("  CONSISTENT: sharp optimum that survives held-out "
                  "recordings. Send the recovered tilt to Mark as a "
                  "cross-check, do NOT treat it as verified.", flush=True)
        else:
            print("  NOT DETERMINED: either the optimum does not generalise "
                  "across recordings or the landscape is a plateau. Ask Mark "
                  "for the tilt angle.", flush=True)
    print("  calibration.yaml NOT modified, by design.", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
