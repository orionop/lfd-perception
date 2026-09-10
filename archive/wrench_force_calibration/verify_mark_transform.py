"""
Score Mark's CAD-derived bota-to-camera transform against the real contact
events, resolving the convention ambiguities empirically.

WHAT ARRIVED (2026-08-24 email, superseding the 13:50 version)

    R = [[-0.70710678,  0.5      , -0.5      ],
         [-0.70710678, -0.5      ,  0.5      ],
         [ 0.        ,  0.70710678, 0.70710678]]
    t = [0, 120.99, 125.65] mm

plus the previously missing piece: the camera axis is 15 degrees off the
183.41 mm diagonal.

TWO THINGS TO RESOLVE FIRST, NEITHER GUESSABLE

  1. SIGN OF tz. His scan reads "z: -125.65"; the typed email says +125.65.
     The same discrepancy appears in his earlier message (-114.64 written,
     114.64 typed). The gap between the two is 251 mm against a 40.2 mm
     tolerance, so it cannot be assumed either way.

  2. DIRECTION AND ROTATION CONVENTION. His R is a valid rotation matrix
     (orthonormal to 3e-9, det +1), but the scipy call he quotes,
     Rotation.from_euler("zxy", [-135, 45, 0]), does NOT reproduce it. And
     "the transformation between force sensor and camera" does not say which
     way it maps. Our pipeline needs T_bota_camera, mapping a camera-frame
     point into the bota frame.

WHY SCORING THESE IS NOT THE SAME AS FITTING

Code/extrinsic_grid_search.py failed honestly: it searched continuous
parameters and overfitted, in-fold 1.000 against held-out 0.067. This script
fits NOTHING. Every candidate is fully determined by Mark's numbers; the only
freedom is a small discrete set of conventions, and each is reported
separately with its hit rate. The grid search already established the null:
across 91,800 candidates the median hit rate was 0.143. A convention that
scores near 1.0 is therefore doing real work, and one near 0.14 is not.

If several conventions score equally well, that is reported as unresolved
rather than papered over.

DOES NOT WRITE calibration.yaml.

Usage:
    .venv_analysis/bin/python Code/verify_mark_transform.py
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_eval_set import build_events
from wrench_ray import pose_to_T, ray_mask_score, ray_pixels, wrench_line_bota

OUT_CSV = "figures/verify_mark_transform.csv"
CALIB = "calibration.yaml"

S_MIN, S_MAX, S_N = -0.60, 0.60, 240

R_MARK = np.array([[-0.70710678,  0.5,        -0.5],
                   [-0.70710678, -0.5,         0.5],
                   [ 0.0,         0.70710678,  0.70710678]])

T_EMAIL2_POS = np.array([0.0, 120.99,  125.65]) / 1000.0
T_EMAIL2_NEG = np.array([0.0, 120.99, -125.65]) / 1000.0
T_EMAIL1_POS = np.array([0.0, 109.97,  114.64]) / 1000.0
T_EMAIL1_NEG = np.array([0.0, 109.97, -114.64]) / 1000.0


def euler_zxy(z_deg, x_deg, y_deg):
    """Reconstruct the scipy call he quotes, both conventions, since lowercase
    axes in scipy mean extrinsic and uppercase mean intrinsic."""
    def Rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def Rx(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def Ry(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    z, x, y = np.radians(z_deg), np.radians(x_deg), np.radians(y_deg)
    extrinsic = Ry(y) @ Rx(x) @ Rz(z)
    intrinsic = Rz(z) @ Rx(x) @ Ry(y)
    return extrinsic, intrinsic


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def load_K():
    import yaml
    with open(CALIB) as f:
        c = yaml.safe_load(f)
    return np.array(c["camera_intrinsics"]["K"], dtype=float)


def score(T_bota_cam, events, K):
    hits, dists, in_frame = [], [], []
    for ev in events:
        r0, fhat = wrench_line_bota(ev["force"], ev["torque"])
        T_base_bota = pose_to_T(*ev["pose"])
        uv, _ = ray_pixels(r0, fhat, T_base_bota, T_bota_cam, K,
                           s_min=S_MIN, s_max=S_MAX, n=S_N)
        hit, d, n_in = ray_mask_score(uv, ev["mask"], ev["W"], ev["H"])
        hits.append(bool(hit))
        dists.append(d)
        in_frame.append(n_in)
    fin = [d for d in dists if np.isfinite(d)]
    return (float(np.mean(hits)), float(np.mean(fin)) if fin else float("inf"),
            int(np.sum(in_frame)), hits)


def main():
    ext, intr = euler_zxy(-135, 45, 0)
    print("[check] Mark's R orthonormality err "
          f"{np.abs(R_MARK @ R_MARK.T - np.eye(3)).max():.2e}, "
          f"det {np.linalg.det(R_MARK):.9f}", flush=True)
    print(f"[check] his quoted scipy euler reproduces his R? "
          f"extrinsic diff {np.abs(ext - R_MARK).max():.3f}, "
          f"intrinsic diff {np.abs(intr - R_MARK).max():.3f}", flush=True)
    print("[check] so the matrix and the euler formula disagree; both are "
          "scored below\n", flush=True)

    events = build_events(verbose=False)
    print(f"[events] {len(events)} contact events across "
          f"{len(set(e['trial'] for e in events))} recordings", flush=True)
    print(f"[null]   grid search over 91,800 candidates had median hit rate "
          f"0.143; that is the bar to beat\n", flush=True)
    K = load_K()

    rot_variants = [
        ("R_mark", R_MARK),
        ("R_mark^T", R_MARK.T),
        ("R_euler_extrinsic", ext),
        ("R_euler_extrinsic^T", ext.T),
        ("R_euler_intrinsic", intr),
        ("R_euler_intrinsic^T", intr.T),
    ]
    trans_variants = [
        ("email2 tz=+125.65", T_EMAIL2_POS),
        ("email2 tz=-125.65", T_EMAIL2_NEG),
        ("email1 tz=+114.64", T_EMAIL1_POS),
        ("email1 tz=-114.64", T_EMAIL1_NEG),
    ]

    rows = []
    print(f"{'rotation':22s} {'translation':20s} {'dir':10s} "
          f"{'hit':>6s} {'meandist':>9s} {'raypx':>7s}", flush=True)
    print("-" * 82, flush=True)
    for rname, R in rot_variants:
        for tname, t in trans_variants:
            T = make_T(R, t)
            for dname, Tc in (("as-given", T), ("inverted", np.linalg.inv(T))):
                rate, dist, npx, hits = score(Tc, events, K)
                rows.append([rname, tname, dname, f"{rate:.4f}",
                             f"{dist:.2f}", npx,
                             "".join("1" if h else "0" for h in hits)])
                flag = "   <<<" if rate >= 0.85 else ""
                print(f"{rname:22s} {tname:20s} {dname:10s} "
                      f"{rate:6.3f} {dist:9.1f} {npx:7d}{flag}", flush=True)

    best = max(rows, key=lambda r: (float(r[3]), -float(r[4])))
    top = [r for r in rows if float(r[3]) >= float(best[3]) - 1e-9]
    print(f"\n[best] hit rate {best[3]}  mean centroid dist {best[4]} px", flush=True)
    print(f"[best] {best[0]}  |  {best[1]}  |  {best[2]}", flush=True)
    print(f"[ties] {len(top)} convention(s) reach that score", flush=True)
    for r in top:
        print(f"        {r[0]:22s} {r[1]:20s} {r[2]}", flush=True)

    print("\n[verdict]", flush=True)
    if float(best[3]) >= 0.85 and len(top) == 1:
        print("  RESOLVED: one convention clearly explains every contact "
              "event. This is independent of our data in the sense that "
              "nothing was fitted; only a discrete convention was chosen.",
              flush=True)
    elif float(best[3]) >= 0.85:
        print(f"  PARTIAL: {len(top)} conventions score equally. Mark must "
              f"disambiguate, or the tie must be broken by geometry rather "
              f"than by these events.", flush=True)
    else:
        print("  REJECTED: no interpretation of the supplied transform puts "
              "the wrench ray on the contact object. Report back to Mark with "
              "the per-event detail rather than guessing further.", flush=True)
    print("  calibration.yaml NOT modified, by design.", flush=True)

    os.makedirs("figures", exist_ok=True)
    if os.path.exists(OUT_CSV):
        import shutil
        shutil.copy2(OUT_CSV, OUT_CSV + ".bak")
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rotation", "translation", "direction", "hit_rate",
                    "mean_centroid_dist_px", "total_ray_px_in_frame",
                    "per_event_hits"])
        w.writerows(rows)
    print(f"[write] {OUT_CSV}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
