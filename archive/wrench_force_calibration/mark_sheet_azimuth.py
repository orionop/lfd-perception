"""
The one thing Mark's side-view drawing cannot fix: azimuth about the tool axis.

CONTEXT
-------
Code/mark_sheet_decode.py shows that exactly one construction of his numbers
reproduces his drawing, to 0.00 mm:

    R = Rz(-135) @ Rx(45)              his intended INTRINSIC composition
    t = R @ [0, 120.99, -125.65] mm    his vector rotated out of the "rotated
                                       frame" it is labelled with
      = [123.32, -123.32, -3.30] mm    camera 174.40 mm out, 3.30 mm above

That construction aims the optical axis 13.4 deg off the gripper centre, the
best of any candidate. But it hits only 3 of 7 contact events, against 6 of 7
for the transform calibration.yaml currently carries -- which in turn puts the
camera 133.5 mm from where his drawing puts it, against a 40.2 mm tolerance.

THE FREE PARAMETER
------------------
His sheet is a side elevation. It fixes the camera's radius from the tool
axis, its height, and its tilt. It does NOT fix which way the bracket points
in the horizontal plane, because that direction is normal to the page. Any
rotation about the measurement frame's z axis,

    T(phi) = Rz(phi) @ T_drawing

leaves every dimension he drew invariant: same radius, same height, same angle
between optical axis and gripper centre. So phi is a genuine unknown of his
derivation rather than a fitted knob, and the recordings are the only thing
that can pin it.

OVERFITTING GUARD
-----------------
One parameter, seven events, three recordings. Code/extrinsic_grid_search.py
already showed that a large free search on this eval set overfits badly
(in-fold 1.000, held-out 0.067), so the sweep is scored two ways: on all seven
events, and leave-one-RECORDING-out. A phi that only survives in-fold is
reported as a failure, not as an answer.

Read only. Writes a CSV and a figure; does not touch calibration.yaml.

Usage: .venv_analysis/bin/python Code/mark_sheet_azimuth.py
"""
import csv
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_eval_set import build_events
from verify_mark_transform import load_K, make_T, score

OUT_CSV = "figures/mark_sheet_azimuth.csv"
OUT_PNG = "figures/mark_sheet_azimuth.png"
STEP_DEG = 1.0

Rz = lambda a: Rot.from_euler("z", a, degrees=True).as_matrix()
Rx = lambda a: Rot.from_euler("x", a, degrees=True).as_matrix()

T_ROT = np.array([0.0, 120.99, -125.65])
D_GRIPPER = 104.0


def drawing_T():
    R = Rz(-135) @ Rx(45)
    return make_T(R, (R @ T_ROT) / 1000.0)


def spin(T0, phi):
    T = np.eye(4)
    T[:3, :3] = Rz(phi) @ T0[:3, :3]
    T[:3, 3] = Rz(phi) @ T0[:3, 3]
    return T


def aim_deg(T):
    t = T[:3, 3] * 1000.0
    v = np.array([0.0, 0.0, D_GRIPPER]) - t
    v /= np.linalg.norm(v)
    return float(np.degrees(np.arccos(np.clip(T[:3, 2] @ v, -1, 1))))


def main():
    K = load_K()
    events = build_events(verbose=False)
    recs = sorted({e["trial"] for e in events})
    print(f"[events] {len(events)} events across {len(recs)} recordings",
          flush=True)
    for r in recs:
        print(f"           {r}: {sum(1 for e in events if e['trial']==r)}",
              flush=True)

    T0 = drawing_T()
    r0 = score(T0, events, K)
    print(f"\n[base]   drawing-consistent T: camera "
          f"{(T0[:3,3]*1000).round(2)} mm, aim {aim_deg(T0):.1f} deg, "
          f"hit {r0[0]:.3f}", flush=True)
    print(f"[sweep]  phi 0..360 step {STEP_DEG} deg; radius, height and tilt "
          f"are invariant under phi\n", flush=True)

    phis = np.arange(0.0, 360.0, STEP_DEG)
    rows = []
    for phi in phis:
        T = spin(T0, phi)
        rate, dist, npx, hits = score(T, events, K)
        rows.append((float(phi), rate, dist, npx, hits))
        if rate >= 0.70:
            print(f"  phi {phi:6.1f}  hit {rate:.3f}  dist {dist:7.1f} px  "
                  f"aim {aim_deg(T):5.1f}  "
                  f"{''.join('1' if h else '0' for h in hits)}", flush=True)

    best = max(rows, key=lambda r: (r[1], -r[2]))
    top = [r for r in rows if r[1] >= best[1] - 1e-9]
    print(f"\n[best]   phi {best[0]:.1f} deg  hit {best[1]:.3f}  "
          f"dist {best[2]:.1f} px", flush=True)
    print(f"[best]   {len(top)}/{len(rows)} angles reach it, "
          f"phi {min(r[0] for r in top):.0f}..{max(r[0] for r in top):.0f} deg",
          flush=True)

    print("\n[held-out] leave one RECORDING out: pick phi on the other two, "
          "score on the held-out one", flush=True)
    ho = []
    for r in recs:
        tr = [e for e in events if e["trial"] != r]
        te = [e for e in events if e["trial"] == r]
        scored = [(score(spin(T0, p), tr, K)[0], -score(spin(T0, p), tr, K)[1],
                   p) for p in phis]
        pick = max(scored)[2]
        s_te = score(spin(T0, pick), te, K)
        ho.append(s_te[0])
        print(f"  hold out {r:22s} n={len(te)}  picked phi {pick:6.1f}  "
              f"held-out hit {s_te[0]:.3f}", flush=True)
    print(f"  mean held-out hit rate {np.mean(ho):.3f}   (chance 0.143)",
          flush=True)

    os.makedirs("figures", exist_ok=True)
    for p in (OUT_CSV, OUT_PNG):
        if os.path.exists(p):
            import shutil
            shutil.copy2(p, p + ".bak")
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phi_deg", "hit_rate", "mean_dist_px", "ray_px_in_frame",
                    "per_event_hits"])
        for phi, rate, dist, npx, hits in rows:
            w.writerow([f"{phi:.1f}", f"{rate:.4f}", f"{dist:.2f}", npx,
                        "".join("1" if h else "0" for h in hits)])
    print(f"[write] {OUT_CSV}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot([r[0] for r in rows], [r[1] for r in rows], lw=1.2)
    ax.axhline(0.143, ls="--", c="gray", lw=1, label="chance (0.143)")
    ax.axhline(6 / 7, ls="--", c="tab:red", lw=1,
               label="adopted extrinsic T (6/7)")
    ax.axvline(0.0, ls=":", c="tab:green", lw=1.2,
               label="phi=0, drawing as read")
    ax.set_xlabel("azimuth phi about the measurement z axis (deg)")
    ax.set_ylabel("wrench-ray hit rate")
    ax.set_title("Azimuth is the one dimension the side-view drawing does not "
                 "fix", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"[write] {OUT_PNG}", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
