"""
Full audit of the bota->camera extrinsic, after repeated confusion about
whose convention is right.

Structure, in order:
  PART A  Mark's own claims, checked as pure algebra (no data).
  PART B  Mark's INTERNAL consistency: his rotation against his own two
          translation vectors, which he gives in both frames. This needs
          none of our recordings and is the strongest check on his sheet.
  PART C  Every plausible way to build T from his (R, t) -- 2 rotations x 6
          translation placements x 2 directions -- each scored with the SAME
          wrench-ray metric the production code uses.
  PART D  Our own pipeline's self-consistency: does the production path use
          T the same way this scorer does?
  PART E  Multiple-comparison check on the winner.

Read only. Prints; writes nothing.

Usage: .venv_analysis/bin/python Code/audit_extrinsic.py
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_mark_transform import (R_MARK, euler_zxy, load_K, make_T,
                                   score)
from contact_eval_set import build_events

Rz = lambda a: Rot.from_euler("z", a, degrees=True).as_matrix()
Rx = lambda a: Rot.from_euler("x", a, degrees=True).as_matrix()
ok = lambda a, b: bool(np.allclose(a, b, atol=1e-9))


def ang(A, B):
    return float(np.degrees(np.arccos(
        np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1))))


# His sheet, verbatim.
T_ROT = np.array([0.0, 120.99, -125.65])          # "In rotated frame"
T_GLOB = np.array([158.82 + 15.58, 3.3, 0.0])     # "In global"


def part_a():
    print("=" * 72)
    print("PART A -- Mark's claims as pure algebra")
    print("=" * 72)
    ext, intr = euler_zxy(-135, 45, 0)
    checks = [
        ("his handwritten R == intrinsic ZXY [-135,45,0]", ok(R_MARK, intr)),
        ("intrinsic ZXY == Rz(-135) @ Rx(45)", ok(intr, Rz(-135) @ Rx(45))),
        ("intrinsic ZXY == extrinsic yxz [0,45,-135]",
         ok(intr, Rot.from_euler("yxz", [0, 45, -135], degrees=True).as_matrix())),
        ("[Rz(-135)Rx(45)]^T == Rx(-45) @ Rz(135)",
         ok((Rz(-135) @ Rx(45)).T, Rx(-45) @ Rz(135))),
        ("his R is a proper rotation (det +1, orthonormal)",
         ok(R_MARK @ R_MARK.T, np.eye(3)) and abs(np.linalg.det(R_MARK) - 1) < 1e-6),
    ]
    for name, v in checks:
        print(f"  [{'OK ' if v else 'BAD'}] {name}")
    print(f"\n  what we use, ext = Rx(45)Rz(-135), vs his R    : "
          f"{ang(ext, R_MARK):6.2f} deg apart")
    print(f"  what we use vs his expectation Rx(-45)Rz(135)  : "
          f"{ang(ext, Rx(-45) @ Rz(135)):6.2f} deg apart")
    print(f"  our meas->cam (ext^T) vs his Rx(-45)Rz(135)    : "
          f"{ang(ext.T, Rx(-45) @ Rz(135)):6.2f} deg apart")
    return ext


def part_b():
    print("\n" + "=" * 72)
    print("PART B -- Mark's sheet against itself (no recordings involved)")
    print("=" * 72)
    print(f"  his 'In global'       t = {T_GLOB.round(2)}  |t| = "
          f"{np.linalg.norm(T_GLOB):.3f} mm")
    print(f"  his 'In rotated frame' t = {T_ROT.round(2)}  |t| = "
          f"{np.linalg.norm(T_ROT):.3f} mm")
    print(f"  magnitudes agree to {abs(np.linalg.norm(T_GLOB)-np.linalg.norm(T_ROT)):.4f}"
          " mm -> they are the same physical vector in two frames\n")
    cands = {"R_mark @ t_glob": R_MARK @ T_GLOB,
             "R_mark^T @ t_glob": R_MARK.T @ T_GLOB,
             "R_mark @ t_rot": R_MARK @ T_ROT,
             "R_mark^T @ t_rot": R_MARK.T @ T_ROT}
    print("  does his own R carry one of his vectors onto the other?")
    for k, v in cands.items():
        tgt = T_ROT if "t_glob" in k else T_GLOB
        err = float(np.linalg.norm(v - tgt))
        print(f"    {k:20s} = {v.round(2)}  vs {tgt.round(2)}  "
              f"err {err:7.2f} mm  {'MATCH' if err < 1.0 else ''}")
    # what rotation WOULD do it
    a = T_GLOB / np.linalg.norm(T_GLOB)
    b = T_ROT / np.linalg.norm(T_ROT)
    print(f"\n  angle between his two vectors: "
          f"{np.degrees(np.arccos(np.clip(a @ b, -1, 1))):.2f} deg")
    for nm, R in (("his R", R_MARK), ("his R^T", R_MARK.T)):
        rot = R @ a
        print(f"    {nm} maps t_glob_hat to {rot.round(3)}; target "
              f"{b.round(3)}; off by "
              f"{np.degrees(np.arccos(np.clip(rot @ b, -1, 1))):.2f} deg")


def part_c(ext):
    print("\n" + "=" * 72)
    print("PART C -- every plausible T built from his numbers, scored on data")
    print("=" * 72)
    K = load_K()
    events = build_events(verbose=False)
    print(f"  {len(events)} contact events\n")
    rots = [("R_mark", R_MARK), ("R_mark^T", R_MARK.T),
            ("ext=Rx(45)Rz(-135)", ext), ("ext^T", ext.T)]
    tvars = [("t_rot", T_ROT), ("-t_rot", -T_ROT),
             ("t_rot z-flip", T_ROT * np.array([1, 1, -1.0]))]
    print(f"  {'rotation':20s} {'translation':14s} {'placement':16s} "
          f"{'hit':>6s} {'dist px':>9s}")
    rows = []
    for rn, R in rots:
        for tn, t in tvars:
            places = [("[R|t]", make_T(R, t / 1000.0)),
                      ("[R|-R t]", make_T(R, -R @ (t / 1000.0))),
                      ("[R|R t]", make_T(R, R @ (t / 1000.0)))]
            for pn, T in places:
                for dn, Tc in (("", T), (" inv", np.linalg.inv(T))):
                    rate, dist, npx, _ = score(Tc, events, K)
                    rows.append((rate, dist, rn, tn, pn + dn))
                    flag = "  <<<" if rate >= 0.85 else ""
                    print(f"  {rn:20s} {tn:14s} {pn+dn:16s} "
                          f"{rate:6.3f} {dist:9.1f}{flag}")
    rows.sort(key=lambda r: (-r[0], r[1]))
    print(f"\n  best: {rows[0][2]} | {rows[0][3]} | {rows[0][4]}  "
          f"hit {rows[0][0]:.3f}  dist {rows[0][1]:.1f} px")
    top = [r for r in rows if r[0] >= rows[0][0] - 1e-9]
    print(f"  {len(top)} construction(s) tie at the top:")
    for r in top:
        print(f"     {r[2]:20s} {r[3]:14s} {r[4]}")
    print(f"  total constructions scored: {len(rows)}")
    return rows


def part_d(ext):
    print("\n" + "=" * 72)
    print("PART D -- does our production path use T the same way?")
    print("=" * 72)
    from geometric_seed import load_calib, project_bota_point
    from wrench_ray import pose_to_T, project_points
    K, T = load_calib()
    print(f"  calibration.yaml R == ext ? {ok(T[:3,:3], ext)}")
    print(f"  calibration.yaml t (mm)   = {(T[:3,3]*1000).round(2)}  "
          f"== his t_rot ? {ok(T[:3,3]*1000, T_ROT)}")
    # the production projector and the audit scorer must agree on direction
    pose = np.array([0.3, 0.1, 0.5, 0.0, 0.0, 0.0, 1.0])
    p_bota = np.array([0.0, 0.0, 0.104])
    uv1, z1 = project_bota_point(p_bota, pose, K, T)
    Tb = pose_to_T(*pose)
    uv2, z2 = project_points((Tb @ np.append(p_bota, 1.0))[:3][None, :],
                             K, Tb @ T)
    print(f"  project_bota_point   -> uv {uv1.round(3)}  z {z1:.4f}")
    print(f"  raw project_points   -> uv {uv2[0].round(3)}  z {z2[0]:.4f}")
    print(f"  agree ? {ok(uv1, uv2[0])}")
    # and confirm T is used as camera->measurement, i.e. inv(T) maps meas->cam
    p_cam = np.linalg.inv(T) @ np.append(p_bota, 1.0)
    print(f"\n  a point 104 mm along measurement +z sits at "
          f"{p_cam[:3].round(4)} m in the CAMERA frame")
    print(f"  its camera-frame depth is {p_cam[2]:.4f} m "
          f"({'in front of' if p_cam[2] > 0 else 'BEHIND'} the camera)")


def part_e(rows):
    print("\n" + "=" * 72)
    print("PART E -- is the winner beating chance, given how many we tried?")
    print("=" * 72)
    n = len(rows)
    best = rows[0][0]
    p = 0.143                      # null hit rate from the 91,800-candidate grid
    k = int(round(best * 7))
    from math import comb
    p_one = sum(comb(7, i) * p**i * (1-p)**(7-i) for i in range(k, 8))
    print(f"  winner {k}/7 events; null per-event hit rate {p}")
    print(f"  P(>= {k}/7 by chance, one try)      = {p_one:.3e}")
    print(f"  constructions tried here           = {n}")
    print(f"  P(any of {n} reaching it by chance) = {1-(1-p_one)**n:.3e}")


if __name__ == "__main__":
    e = part_a()
    part_b()
    r = part_c(e)
    part_d(e)
    part_e(r)
    print("\n[done]")
