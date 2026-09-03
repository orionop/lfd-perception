"""
Recover the tool-axis direction in the bota frame, under a hard constraint.

THE PROBLEM
-----------
Mark's drawing pins the gripper centre at 104 mm from the bota origin along
the tool axis. That decodes exactly: 104 - 91.72 = 12.28 matches his "12.28"
dimension, and -12.28 + 15.58 = 3.30 matches his stated global "y: 3.3".

What it does not give is the DIRECTION of that 104 mm in the bota frame,
because the rotation from his drawing-global frame to the bota frame is the
one step his scan omits. Code/verify_mark_transform.py confirmed no standard
Euler convention, transpose or signed axis permutation reproduces his own
global-to-rotated conversion (closest 51 mm off), so it cannot be inferred
from his numbers alone. Testing the six signed axes failed: none lands inside
the grasped object, best 250.7 px away.

THE CONSTRAINT THAT MAKES THIS SOLVABLE
---------------------------------------
The unknown rotation R (drawing-global -> bota) is not free. It must map his
global camera vector onto the verified bota camera vector:

    R @ [174.40, 3.3, 0]  =  [0, 120.99, -125.65]        (millimetres)

Both are known: the first from his drawing, the second from calibration.yaml
after validation at 6/7 contact events. A single vector correspondence pins
every degree of freedom except one -- rotation about the shared direction.

So the search space is ONE parameter, not three, and it is bounded by
geometry rather than by guesswork.

WHY THIS IS NOT THE EARLIER OVERFIT
-----------------------------------
Code/extrinsic_grid_search.py searched four free parameters over 91,800
candidates and overfitted badly, in-fold 1.000 against held-out 0.067. This
searches one parameter under a hard constraint, and is validated
leave-one-trial-out across five independent grasp recordings. If the winner
does not survive held-out trials, that is reported as a failure and the
grasped role stays unseeded.

Prints for review. Writes the derived offset to a small JSON that
Code/geometric_seed.py reads. Does NOT modify calibration.yaml.

Usage:
    .venv_analysis/bin/python Code/derive_tool_axis.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from geometric_seed import (GRIPPER_OFFSET_MM, grasp_events, load_calib,
                            project_bota_point)

OUT_JSON = "figures/tool_axis.json"

# From Mark's drawing, millimetres, in his drawing-global frame with the bota
# origin at the origin and y up.
G_CAM = np.array([158.82 + 15.58, 3.3, 0.0])
G_GRIPPER = np.array([0.0, -104.0, 0.0])


def align(a, b):
    """Rotation taking unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def rot_about(axis, deg):
    k = axis / np.linalg.norm(axis)
    t = np.radians(deg)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def score(offset_m, evs, K, T_bc):
    inm, dists = 0, []
    for e in evs:
        uv, z = project_bota_point(offset_m, e["pose"], K, T_bc)
        H, W = e["mask"].shape
        if z <= 0 or not (0 <= uv[0] < W and 0 <= uv[1] < H):
            dists.append(1e6)
            continue
        if e["mask"][int(uv[1]), int(uv[0])]:
            inm += 1
        ys, xs = np.nonzero(e["mask"])
        dists.append(float(np.hypot(uv[0] - xs.mean(), uv[1] - ys.mean())))
    return inm, float(np.mean(dists)), dists


def main():
    K, T_bc = load_calib()
    t_cam = T_bc[:3, 3] * 1000.0
    print(f"[constraint] his global camera vector {G_CAM.round(2).tolist()} mm"
          f"  |.| {np.linalg.norm(G_CAM):.3f}", flush=True)
    print(f"[constraint] verified bota camera vector "
          f"{t_cam.round(2).tolist()} mm  |.| {np.linalg.norm(t_cam):.3f}",
          flush=True)
    dm = abs(np.linalg.norm(G_CAM) - np.linalg.norm(t_cam))
    print(f"[constraint] magnitudes differ by {dm:.4f} mm, so one rotation "
          f"genuinely relates them", flush=True)
    if dm > 1.0:
        print("[fatal] magnitudes disagree; the two vectors are not the same "
              "physical offset", flush=True)
        sys.exit(1)

    R0 = align(G_CAM, t_cam)
    axis = t_cam / np.linalg.norm(t_cam)
    print(f"[family] one free parameter: rotation about the shared direction\n",
          flush=True)

    evs = grasp_events(K, T_bc)
    trials = [e["trial"] for e in evs]
    print(f"[events] {len(evs)} grasp events: {trials}\n", flush=True)
    if len(evs) < 3:
        print("[fatal] too few grasp events to cross-validate", flush=True)
        sys.exit(1)

    thetas = np.arange(0.0, 360.0, 0.25)
    rows = []
    for th in thetas:
        R = rot_about(axis, th) @ R0
        off = (R @ G_GRIPPER) / 1000.0
        inm, md, dists = score(off, evs, K, T_bc)
        rows.append((th, inm, md, off, dists))

    inms = np.array([r[1] for r in rows])
    mds = np.array([r[2] for r in rows])
    order = np.lexsort((mds, -inms))
    best = rows[order[0]]
    print(f"[fit-all] best theta {best[0]:.2f} deg  -> "
          f"{best[1]}/{len(evs)} inside the grasped mask, "
          f"mean {best[2]:.1f} px", flush=True)
    print(f"[fit-all] gripper offset in bota frame = "
          f"{(best[3]*1000).round(2).tolist()} mm", flush=True)
    print(f"[null]    over the {len(thetas)} candidate angles: median "
          f"{np.median(inms):.1f}/{len(evs)} in-mask, best {inms.max()}",
          flush=True)

    print("\n[loto] leave-one-trial-out", flush=True)
    held_ok = 0
    for h in range(len(evs)):
        keep = [i for i in range(len(evs)) if i != h]
        sub = [(r[0], sum(1 for i in keep if r[4][i] < 1e5 and _inside(evs[i], r[3], K, T_bc)),
                float(np.mean([r[4][i] for i in keep]))) for r in rows]
        sinm = np.array([s[1] for s in sub])
        smd = np.array([s[2] for s in sub])
        b = sub[int(np.lexsort((smd, -sinm))[0])]
        Rb = rot_about(axis, b[0]) @ R0
        offb = (Rb @ G_GRIPPER) / 1000.0
        ok = _inside(evs[h], offb, K, T_bc)
        held_ok += int(ok)
        print(f"  held out {evs[h]['trial']:24s} theta {b[0]:7.2f} -> "
              f"{'IN ' if ok else 'out'}", flush=True)

    print(f"\n[loto] {held_ok}/{len(evs)} held-out trials land inside the "
          f"grasped object", flush=True)

    if best[1] >= max(3, int(0.6 * len(evs))) and held_ok >= int(0.6 * len(evs)):
        os.makedirs("figures", exist_ok=True)
        with open(OUT_JSON, "w") as f:
            json.dump({"theta_deg": float(best[0]),
                       "offset_bota_m": (best[3]).tolist(),
                       "in_mask": int(best[1]), "n_events": len(evs),
                       "loto_in_mask": int(held_ok),
                       "mean_dist_px": float(best[2])}, f, indent=2)
        print(f"[write] {OUT_JSON}", flush=True)
        print("[verdict] tool axis RECOVERED and survives held-out trials.",
              flush=True)
    else:
        print("[verdict] NOT recovered. Either no angle explains the grasp "
              "events or it fails held-out. Grasped role stays unseeded; wait "
              "for Mark's confirmation of the convention.", flush=True)
        if os.path.exists(OUT_JSON):
            os.rename(OUT_JSON, OUT_JSON + ".rejected")
            print(f"[note] previous {OUT_JSON} moved aside as .rejected",
                  flush=True)
    print("[done]", flush=True)


def _inside(ev, off_m, K, T_bc):
    uv, z = project_bota_point(off_m, ev["pose"], K, T_bc)
    H, W = ev["mask"].shape
    if z <= 0 or not (0 <= uv[0] < W and 0 <= uv[1] < H):
        return False
    return bool(ev["mask"][int(uv[1]), int(uv[0])])


if __name__ == "__main__":
    main()
