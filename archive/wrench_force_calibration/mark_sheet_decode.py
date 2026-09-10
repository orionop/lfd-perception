"""
Decode Mark's CAD sheet end to end, and score every construction it admits.

WHY THIS EXISTS
---------------
calibration.yaml adopted T_bota_camera = [R_ext | t] with
    R_ext = Rx(45) @ Rz(-135)          (the EXTRINSIC reading of his angles)
    t     = [0, 120.99, -125.65] mm    (his vector, used as-is)
on the strength of a 6/7 wrench-ray hit rate. Mark intended the INTRINSIC
composition Rz(-135) @ Rx(45), and has now asked, twice, why the other one
works.

His sheet labels that vector "In rotated frame", not "In global". If it is
expressed in the ROTATED (camera) frame rather than in the measurement frame,
then using it directly as the translation of T_bota_camera is wrong, and the
composition-order question may be the wrong question entirely.

THE INDEPENDENT CHECK
---------------------
His drawing fixes the camera position in the measurement frame without any
rotation algebra at all:

    gripper centre        104.00 mm below his origin, on the tool axis
                          (= 12.28 + 91.72, both dimensioned on the sheet)
    CAD origin            91.72 mm above the gripper, so 12.28 mm BELOW his
    lens                  158.82 mm horizontal, level with the CAD origin
                          -> 12.28 mm below his origin
    lens -> camera origin +15.58 mm on both the horizontal and the vertical
                          (his 15.58 at 45 deg)
    camera origin         174.40 mm horizontal, 3.30 mm ABOVE his origin

Mark states z_measurement points DOWN the tool axis. So in the measurement
frame the camera sits at radius 174.40 mm with z = -3.30 mm, and the gripper
centre at [0, 0, +104]. The azimuth is not fixed by the side view; everything
else is.

That gives a pass/fail that uses none of our recordings: any candidate T whose
translation does not put the camera 174.40 mm out and 3.30 mm above the origin
is not describing the rig Mark drew, whatever it scores.

WHAT IS REPORTED
----------------
For every (rotation, translation-placement) pair: the implied camera position,
its disagreement with the drawing, where the optical axis points relative to
the gripper, and the wrench-ray hit rate on the shared 7-event set. Physical
consistency and empirical score are reported side by side and NOT combined,
because the point of the exercise is to find out whether they disagree.

Nothing is fitted. Read only; writes a CSV, does not touch calibration.yaml.

Usage: .venv_analysis/bin/python Code/mark_sheet_decode.py
"""
import csv
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_eval_set import build_events
from verify_mark_transform import load_K, make_T, score

OUT_CSV = "figures/mark_sheet_decode.csv"
FOV_HALF_DEG = 87.0 / 2.0

Rz = lambda a: Rot.from_euler("z", a, degrees=True).as_matrix()
Rx = lambda a: Rot.from_euler("x", a, degrees=True).as_matrix()

# --- his sheet, verbatim -----------------------------------------------------
T_ROT = np.array([0.0, 120.99, -125.65])        # "In rotated frame", mm
T_GLOB = np.array([158.82 + 15.58, 3.30, 0.0])  # "In global", mm
D_GRIPPER = 104.0                               # 12.28 + 91.72, mm

# --- the drawing, decoded without any rotation -------------------------------
DRAW_RADIUS = 158.82 + 15.58        # horizontal, mm
DRAW_Z = -(15.58 - 12.28)           # z DOWN positive -> camera 3.30 mm ABOVE


def physical(T):
    """Camera position + pointing implied by a candidate T_bota_camera."""
    t = T[:3, 3] * 1000.0
    R = T[:3, :3]
    radius = float(np.hypot(t[0], t[1]))
    z = float(t[2])
    d_radius = radius - DRAW_RADIUS
    d_z = z - DRAW_Z
    v = np.array([0.0, 0.0, D_GRIPPER]) - t
    v /= np.linalg.norm(v)
    off = float(np.degrees(np.arccos(np.clip(R[:, 2] @ v, -1, 1))))
    return radius, z, d_radius, d_z, off


def main():
    intr = Rz(-135) @ Rx(45)
    ext = Rx(45) @ Rz(-135)

    print("=" * 78)
    print("PART 1 -- the drawing alone, no rotation algebra")
    print("=" * 78)
    print(f"  gripper centre        [0, 0, {D_GRIPPER:.2f}] mm  (z points down)")
    print(f"  camera radius         {DRAW_RADIUS:.2f} mm from the tool axis")
    print(f"  camera height         {DRAW_Z:+.2f} mm in z  "
          f"(i.e. {abs(DRAW_Z):.2f} mm ABOVE his origin)")
    print(f"  |camera - origin|     {np.hypot(DRAW_RADIUS, DRAW_Z):.3f} mm")
    print(f"  |camera - gripper|    "
          f"{np.hypot(DRAW_RADIUS, D_GRIPPER - DRAW_Z):.3f} mm")
    print(f"\n  his 'In global'  t = {T_GLOB.round(2)}  |t| = "
          f"{np.linalg.norm(T_GLOB):.3f} mm")
    print(f"  his 'In rotated' t = {T_ROT.round(2)}  |t| = "
          f"{np.linalg.norm(T_ROT):.3f} mm")
    print("  same length, so the two are one vector written in two frames.")

    print("\n" + "=" * 78)
    print("PART 2 -- which placement of his vector matches the drawing?")
    print("=" * 78)
    print(f"  target: radius {DRAW_RADIUS:.2f} mm, z {DRAW_Z:+.2f} mm\n")
    print(f"  {'construction':28s} {'camera pos (mm)':30s} "
          f"{'radius':>8s} {'z':>8s} {'err':>8s}")
    for nm, R in (("intr", intr), ("ext", ext)):
        for pn, tv in ((f"t_rot", T_ROT),
                       (f"{nm} @ t_rot", R @ T_ROT),
                       (f"{nm}^T @ t_rot", R.T @ T_ROT)):
            r = float(np.hypot(tv[0], tv[1]))
            err = np.hypot(r - DRAW_RADIUS, tv[2] - DRAW_Z)
            flag = "   <== matches drawing" if err < 1.0 else ""
            print(f"  {pn:28s} {str(tv.round(2)):30s} "
                  f"{r:8.2f} {tv[2]:8.2f} {err:8.2f}{flag}")

    print("\n" + "=" * 78)
    print("PART 3 -- every construction: physical consistency AND wrench score")
    print("=" * 78)
    K = load_K()
    events = build_events(verbose=False)
    print(f"  {len(events)} contact events; chance hit rate 0.143\n")

    cands = []
    for rn, R in (("intr = Rz(-135)Rx(45)", intr), ("intr^T", intr.T),
                  ("ext  = Rx(45)Rz(-135)", ext), ("ext^T", ext.T)):
        for pn, tv in (("t_rot as-is", T_ROT),
                       ("R @ t_rot", R @ T_ROT),
                       ("R^T @ t_rot", R.T @ T_ROT),
                       ("t_glob as-is", T_GLOB)):
            cands.append((rn, pn, make_T(R, tv / 1000.0)))

    print(f"  {'rotation':22s} {'translation':14s} {'rad':>7s} {'z':>7s} "
          f"{'drawErr':>8s} {'aim':>7s} {'inFOV':>6s} {'hit':>6s} {'px':>7s}")
    rows = []
    for rn, pn, T in cands:
        radius, z, dr, dz, off = physical(T)
        derr = float(np.hypot(dr, dz))
        rate, dist, npx, hits = score(T, events, K)
        rows.append([rn, pn, f"{radius:.2f}", f"{z:.2f}", f"{derr:.2f}",
                     f"{off:.1f}", off < FOV_HALF_DEG, f"{rate:.3f}",
                     f"{dist:.1f}", "".join("1" if h else "0" for h in hits)])
        print(f"  {rn:22s} {pn:14s} {radius:7.2f} {z:7.2f} {derr:8.2f} "
              f"{off:7.1f} {str(off < FOV_HALF_DEG):>6s} {rate:6.3f} "
              f"{dist:7.1f}")

    print("\n" + "=" * 78)
    print("PART 4 -- do physics and score agree?")
    print("=" * 78)
    phys = [r for r in rows if float(r[4]) < 1.0]
    best_score = max(rows, key=lambda r: (float(r[7]), -float(r[8])))
    print(f"  constructions matching the drawing (< 1 mm): {len(phys)}")
    for r in phys:
        print(f"     {r[0]:22s} {r[1]:14s} aim {r[5]:>6s} deg  "
              f"hit {r[7]}  dist {r[8]} px  events {r[9]}")
    print(f"\n  best-scoring construction:")
    print(f"     {best_score[0]:22s} {best_score[1]:14s} "
          f"drawErr {best_score[4]} mm  hit {best_score[7]}  "
          f"dist {best_score[8]} px  events {best_score[9]}")
    if phys and best_score not in phys:
        print("\n  THESE DISAGREE. The construction that reproduces Mark's own")
        print("  drawing is not the one that scores best on the recordings.")
        print("  Neither can be dismissed: the drawing check uses no data, and")
        print("  the score uses no drawing. Report both, adopt neither yet.")
    elif phys and best_score in phys:
        print("\n  THESE AGREE -- the drawing-consistent construction is also")
        print("  the best scoring one.")
    else:
        print("\n  NOTHING matches the drawing. The decode above is wrong, or")
        print("  the measurement frame is not where the sheet places it.")

    os.makedirs("figures", exist_ok=True)
    if os.path.exists(OUT_CSV):
        import shutil
        shutil.copy2(OUT_CSV, OUT_CSV + ".bak")
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rotation", "translation", "radius_mm", "z_mm",
                    "drawing_err_mm", "aim_off_deg", "in_fov", "hit_rate",
                    "mean_dist_px", "per_event_hits"])
        w.writerows(rows)
    print(f"\n[write] {OUT_CSV}")
    print("[done]")


if __name__ == "__main__":
    main()
