"""
How accurate must bota_to_camera be for wrench-ray grounding to work?

The CAD failure (Code/cad_candidate_sensitivity.py: both candidates miss by
551-760 px) left an important question open -- is the method asking for
impossible precision, or were those two particular candidates simply wrong?
This script answers it, and states the accuracy bar the hand-eye calibration
(Code/calibrate_hand_eye.py) has to clear to be worth trusting.

A FIRST ATTEMPT AT THIS WAS WRONG, and the reason is worth recording. It
Monte-Carlo'd perturbations around calibration.yaml's nominal extrinsic and
measured how often the projected ray landed inside the true object's bbox.
That returns 0% even at zero perturbation -- because the nominal IS the known-
bad CAD value. It measures nothing except that the anchor is broken. Anchoring
instead on an extrinsic fitted to make the ray hit would be circular.

The tolerance does not actually need any reference extrinsic. It is set by the
object's angular size at the camera, which is measurable directly:

    a rotation error of theta shifts the projected point by ~f*tan(theta) px
    a translation error of d at depth z shifts it by  ~f*d/z px

so, writing R for the object's smallest half-extent in pixels (the tightest
direction it can be missed in), the ray stays on the object while

    theta < arctan(R / f)          and          d < z * R / f

Both R and z come from real measurements: R from the SAM2-propagated
contact-receiver mask in the delivered sidecar, z from the trial's real depth
map at that mask. f is the lab's own calibrated focal length. Nothing is
simulated and nothing is fitted, so this is a property of the geometry rather
than of any particular calibration guess.

A Monte-Carlo cross-check is also run: perturb a synthetic extrinsic that is
correct by construction (built to place the camera at the measured depth,
looking at the contact point) and confirm the empirical hit-rate falls off
where the closed form predicts. Its only job is to catch algebra errors.

Output: figures/calibration_sensitivity.png + .csv, and a printed requirement.

Run inside .venv_analysis:
    .venv_analysis/bin/python Code/calibration_sensitivity_sweep.py
"""
import ast
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from PIL import Image

POSE = "NS_1.franka_robot_state_broadcaster.current_pose.pose."
W = "bota_post.wrench_body_compensated.wrench."
GRIP = "NS_1.franka_gripper.joint_states.position"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"
DEPTH = "zed.zed_node.depth.depth_registered.compressedDepth"

TRIALS = [
    ("Rec2 (plate)",  "Data/lfdws_t001_depth/lfdws_t001_depth_0.csv",
     "figures/identify_depth",
     "Data/lfdws_t001_depth/zed_zed_node_depth_depth_registered_compressedDepth", "png"),
    ("Rec6 (latch)",  "Data/lfdws_t001_labexport/lfdws_t001/lfdws_t001.csv",
     "figures/t001labexport/identify",
     "Data/lfdws_t001_labexport/lfdws_t001/zed_zed_node_depth_depth_registered_compressedDepth", "npy"),
]

ROT_DEG = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0]
TRANS_MM = [0.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0]
N_DRAWS = 300
SEED = 0
OUT_PNG = "figures/calibration_sensitivity.png"
OUT_CSV = "figures/calibration_sensitivity.csv"


def parse_gw(c):
    try:
        return float(np.sum(ast.literal_eval(c)))
    except Exception:
        return float("nan")


def press_row(df):
    fm = np.sqrt(df[W+"force.x"]**2 + df[W+"force.y"]**2 + df[W+"force.z"]**2).to_numpy()
    base = np.median(fm[:len(fm)//10])
    if GRIP in df.columns:
        w = df[GRIP].apply(parse_gw).to_numpy()
        if float(np.nanmax(w) - np.nanmin(w)) >= 0.001:
            thr = np.nanmin(w) + 0.5*(np.nanmax(w) - np.nanmin(w))
            closed = w < thr
            if closed.any():
                return int(np.argmax(np.where(closed, fm - base, -np.inf)))
    return int(np.argmax(fm - base))


def receiver_mask_geom(sidecar_dir, img_id):
    """Bbox, centroid and smallest half-extent (px) of the propagated
    contact-receiver at the event frame (nearest frame if exact missing)."""
    rows = [r for r in csv.DictReader(open(sidecar_dir + "/objects_summary.csv"))
            if r["role"] == "contact_receiver" and float(r["bbox_x0"]) >= 0]
    if not rows:
        return None
    exact = [r for r in rows if r["img_filename"] == f"{img_id}.png"]
    r = exact[0] if exact else rows[len(rows)//2]
    bb = [float(r["bbox_x0"]), float(r["bbox_y0"]), float(r["bbox_x1"]), float(r["bbox_y1"])]
    half = min((bb[2]-bb[0])/2, (bb[3]-bb[1])/2)
    return bb, ((bb[0]+bb[2])/2, (bb[1]+bb[3])/2), half, r["img_filename"], bool(exact)


def depth_at(depth_dir, depth_id, ext, cen):
    p = os.path.join(depth_dir, f"{depth_id}.{ext}")
    if not os.path.exists(p):
        return None
    d = np.load(p).astype(float) if ext == "npy" else \
        np.array(Image.open(p)).astype(float)/1000.0     # 16-bit mm -> m
    u, v = int(round(cen[0])), int(round(cen[1]))
    win = d[max(0, v-7):v+8, max(0, u-7):u+8]
    win = win[(win > 0.05) & (win < 5.0)]
    return float(np.median(win)) if win.size else None


def main():
    calib = yaml.safe_load(open("calibration.yaml"))
    K = np.array(calib["camera_intrinsics"]["K"], dtype=float)
    f = float(K[0][0])
    print(f"[setup] focal length f = {f:.2f} px (lab-calibrated)\n", flush=True)

    cases = []
    for name, csv_path, side, ddir, ext in TRIALS:
        df = pd.read_csv(csv_path)
        i = press_row(df)
        r = df.iloc[i]
        g = receiver_mask_geom(side, str(r[IMG]))
        if g is None:
            print(f"[skip] {name}: no receiver mask", flush=True); continue
        bb, cen, half, used_img, exact = g
        depth_id = str(r[DEPTH]) if DEPTH in df.columns else str(r[IMG])
        z = depth_at(ddir, depth_id, ext, cen)
        if z is None:
            # depth frame is matched to the event row; fall back to the frame
            # actually used for the mask if that row's depth is unreadable
            z = depth_at(ddir, used_img.replace(".png", ""), ext, cen)
        if z is None:
            print(f"[skip] {name}: no readable depth at the mask", flush=True); continue
        th = np.degrees(np.arctan(half/f))
        dm = z*half/f*1000
        cases.append(dict(name=name, half=half, z=z, th=th, dm=dm, bb=bb, cen=cen))
        print(f"[case] {name}: bbox={[round(x) for x in bb]}  half-extent={half:.0f}px  "
              f"depth={z:.3f}m{'' if exact else '  [nearest-frame mask]'}", flush=True)
        print(f"        -> tolerates {th:.2f} deg rotation  or  {dm:.1f} mm translation",
              flush=True)

    if not cases:
        print("[fatal] no usable cases", flush=True); return

    th_req = min(c["th"] for c in cases)
    dm_req = min(c["dm"] for c in cases)

    # ---- Monte-Carlo cross-check on a by-construction-correct extrinsic ----
    rng = np.random.default_rng(SEED)
    grid = np.zeros((len(ROT_DEG), len(TRANS_MM)))
    rows_out = []
    for ri, rot in enumerate(ROT_DEG):
        for ti, tr in enumerate(TRANS_MM):
            hits = tot = 0
            for c in cases:
                # camera at origin looking down +z at the contact point, placed
                # at the measured depth: correct by construction at zero error
                P = np.array([0.0, 0.0, c["z"]])
                for _ in range(N_DRAWS):
                    Pp = P.copy()
                    if rot > 0:
                        a = rng.normal(size=3); a /= np.linalg.norm(a)
                        Kx = np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
                        R = np.eye(3)+np.sin(np.radians(rot))*Kx+(1-np.cos(np.radians(rot)))*(Kx@Kx)
                        Pp = R @ Pp
                    if tr > 0:
                        d = rng.normal(size=3); d /= np.linalg.norm(d)
                        Pp = Pp + d*(tr/1000.0)
                    tot += 1
                    if Pp[2] <= 1e-6:
                        continue
                    du = f*Pp[0]/Pp[2]; dv = f*Pp[1]/Pp[2]
                    if np.hypot(du, dv) <= c["half"]:
                        hits += 1
            grid[ri, ti] = hits/tot if tot else 0.0
            rows_out.append([rot, tr, hits, tot, f"{grid[ri,ti]:.4f}"])
        print(f"[mc] rot={rot:4.1f}deg  " + "  ".join(
            f"{TRANS_MM[j]:.0f}mm:{grid[ri,j]*100:5.1f}%" for j in range(len(TRANS_MM))),
            flush=True)

    os.makedirs("figures", exist_ok=True)
    with open(OUT_CSV, "w") as fh:
        w = csv.writer(fh)
        w.writerow(["rot_err_deg", "trans_err_mm", "hits", "trials", "hit_rate"])
        w.writerows(rows_out)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    im = ax.imshow(grid*100, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=100,
                   extent=[-0.5, len(TRANS_MM)-0.5, -0.5, len(ROT_DEG)-0.5])
    ax.set_xticks(range(len(TRANS_MM))); ax.set_xticklabels([f"{t:.0f}" for t in TRANS_MM])
    ax.set_yticks(range(len(ROT_DEG)));  ax.set_yticklabels([f"{r:.1f}" for r in ROT_DEG])
    ax.set_xlabel("translation error (mm)"); ax.set_ylabel("rotation error (deg)")
    ax.set_title("Wrench ray stays on the contact object\n"
                 f"(real masks + real depth, {len(cases)} contact events)", fontsize=10)
    for i in range(len(ROT_DEG)):
        for j in range(len(TRANS_MM)):
            ax.text(j, i, f"{grid[i,j]*100:.0f}", ha="center", va="center",
                    color="w" if grid[i, j] < 0.6 else "k", fontsize=8)
    fig.colorbar(im, ax=ax, label="hit rate (%)")
    fig.tight_layout(); fig.savefig(OUT_PNG, dpi=130)

    print(f"\n[write] {OUT_CSV}\n[write] {OUT_PNG}", flush=True)
    print("\n[REQUIREMENT] to keep the wrench ray on the contact object at every"
          " tested event:", flush=True)
    print(f"  rotation error  < {th_req:.2f} deg", flush=True)
    print(f"  translation err < {dm_req:.1f} mm", flush=True)
    print(f"  (binding case: {min(cases, key=lambda c: c['th'])['name']})", flush=True)

    # Put the failed CAD attempt on the same scale: its measured miss distance
    # (Code/cad_candidate_sensitivity.py) expressed as an equivalent angle.
    cad_px = [637, 760, 551, 662, 574, 685]      # both candidates x 3 recordings
    cad_deg = [np.degrees(np.arctan(p/f)) for p in cad_px]
    print(f"\n[context] the CAD-derived extrinsics missed by {min(cad_px)}-{max(cad_px)} px,"
          f" i.e. {min(cad_deg):.1f}-{max(cad_deg):.1f} deg of equivalent", flush=True)
    print(f"          pointing error -- roughly {min(cad_deg)/th_req:.0f}x beyond the"
          f" {th_req:.1f} deg the method tolerates.", flush=True)
    print(f"          So CAD did not fail marginally; it failed by an order of"
          f" magnitude.", flush=True)
    print(f"\n[outlook] the requirement itself is loose: a ChArUco hand-eye solve"
          f" routinely reaches", flush=True)
    print(f"          well under 1 deg and a few mm, comfortably inside"
          f" {th_req:.1f} deg / {dm_req:.0f} mm.", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
