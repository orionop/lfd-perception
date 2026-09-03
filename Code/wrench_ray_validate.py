"""
Score ANY candidate T_bota_camera against the real contact events, and draw it.

This is the harness that runs the moment Mark sends a number. It is
deliberately separate from Code/extrinsic_grid_search.py: the search proposes,
this disposes, and it accepts a transform from any source without caring where
it came from.

Three input routes:

    --from_calib                 read calibration.yaml's bota_to_camera block
    --params psi,tilt,d,lens     the 4 search parameters (see wrench_ray.py)
    --tilt 42.5                  Mark's CAD angle alone, with the other three
                                 taken from the CAD translation and defaults

The go/no-go test the paper hangs on: at each contact event, project the
measured wrench line (Bicchi 1990) into the image and check whether it lands
on the object that actually received the contact. This is the check that
Code/project_ee.py's docstring has described as the pre-test since the rig
model was corrected, now runnable.

Draws one panel per event to figures/wrench_ray_validation.png: the ray in
red, the ground-truth mask outlined in its role colour, hit or miss in the
caption. Writes per-event numbers to CSV.

DOES NOT WRITE calibration.yaml.

Usage:
    .venv_analysis/bin/python Code/wrench_ray_validate.py --from_calib
    .venv_analysis/bin/python Code/wrench_ray_validate.py --params 90,45,120,0
    .venv_analysis/bin/python Code/wrench_ray_validate.py --tilt 30
"""
import argparse
import csv
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_eval_set import COLOR, build_events
from wrench_ray import (make_T_bota_camera, pose_to_T, ray_mask_score,
                        ray_pixels, wrench_line_bota)

CALIB = "calibration.yaml"
OUT_PNG = "figures/wrench_ray_validation.png"
OUT_CSV = "figures/wrench_ray_validation.csv"

S_MIN, S_MAX, S_N = -0.60, 0.60, 240


def backup_if_exists(path):
    if os.path.exists(path):
        shutil.copy2(path, path + ".bak")
        print(f"[backup] {path} -> {path}.bak", flush=True)


def load_K_and_T(args):
    import yaml
    with open(CALIB) as f:
        c = yaml.safe_load(f)
    intr = c["camera_intrinsics"]
    if not intr.get("filled", False):
        print("[fatal] camera_intrinsics is not filled:true", flush=True)
        sys.exit(1)
    K = np.array(intr["K"], dtype=float)

    if args.from_calib:
        blk = c.get("bota_to_camera", {})
        T = np.array(blk["T"], dtype=float)
        note = ("calibration.yaml bota_to_camera"
                f" (filled={blk.get('filled', False)})")
        if not blk.get("filled", False):
            print("[warn] bota_to_camera is filled:false, this is the "
                  "documented PRELIMINARY CAD estimate. Scoring it anyway "
                  "because that is the point of this script.", flush=True)
        return K, T, note

    if args.raw_R and args.raw_t_mm:
        R = np.array([float(x) for x in args.raw_R.split(",")]).reshape(3, 3)
        tv = np.array([float(x) for x in args.raw_t_mm.split(",")]) / 1000.0
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, tv
        return K, T, (f"raw R + t={np.round(tv*1000, 2).tolist()} mm "
                      f"(supplied externally, nothing fitted)")

    if args.params:
        psi, tilt, d, lens = [float(x) for x in args.params.split(",")]
    else:
        psi, tilt, d, lens = args.psi, args.tilt, args.d_axial, args.lens_shift
    T = make_T_bota_camera(psi, tilt, d, lens)
    note = (f"psi={psi:.2f} tilt={tilt:.2f} d_axial={d:.2f} "
            f"lens_shift={lens:+.2f}")
    return K, T, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from_calib", action="store_true",
                    help="score calibration.yaml's bota_to_camera as-is")
    ap.add_argument("--params", default=None,
                    help="psi,tilt,d_axial,lens_shift")
    ap.add_argument("--raw_R", default=None,
                    help="9 comma-separated values, row-major rotation")
    ap.add_argument("--raw_t_mm", default=None,
                    help="3 comma-separated translation values, millimetres")
    ap.add_argument("--psi", type=float, default=90.0)
    ap.add_argument("--tilt", type=float, default=30.0)
    ap.add_argument("--d_axial", type=float, default=104.0)
    ap.add_argument("--lens_shift", type=float, default=0.0)
    ap.add_argument("--out_png", default=OUT_PNG)
    ap.add_argument("--out_csv", default=OUT_CSV)
    args = ap.parse_args()

    K, T_bota_cam, note = load_K_and_T(args)
    print(f"[calib] scoring: {note}", flush=True)
    print(f"[calib] T_bota_camera translation = "
          f"{(T_bota_cam[:3,3]*1000).round(2).tolist()} mm", flush=True)

    events = build_events()
    if not events:
        print("[fatal] no usable contact events", flush=True)
        sys.exit(1)

    rows, panels = [], []
    n_hit = 0
    for ev in events:
        r0, fhat = wrench_line_bota(ev["force"], ev["torque"])
        orth = abs(float(r0 @ fhat))
        T_base_bota = pose_to_T(*ev["pose"])
        uv, z = ray_pixels(r0, fhat, T_base_bota, T_bota_cam, K,
                           s_min=S_MIN, s_max=S_MAX, n=S_N)
        hit, dist, n_in = ray_mask_score(uv, ev["mask"], ev["W"], ev["H"])
        n_hit += int(hit)
        print(f"[event] {ev['trial']}/{ev['event']:20s} "
              f"|F|={ev['force_mag']:6.2f} N  "
              f"ray pixels in frame={n_in:4d}  "
              f"{'HIT ' if hit else 'MISS'}  "
              f"dist to mask centroid={dist:8.1f} px  "
              f"(|r0.fhat|={orth:.2e})", flush=True)
        rows.append([ev["trial"], ev["event"], ev["role"], ev["img_id"],
                     f"{ev['force_mag']:.3f}", int(hit), f"{dist:.2f}",
                     n_in, ev["mask_source"], f"{ev['mask_frac']:.5f}"])

        img = cv2.imread(ev["rgb_path"])
        if img is None:
            continue
        col = COLOR[ev["role"]]
        cont, _ = cv2.findContours(ev["mask"].astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cont, -1, col, 2)
        inb = [(int(u), int(v)) for u, v in uv
               if 0 <= u < ev["W"] and 0 <= v < ev["H"]]
        for k in range(1, len(inb)):
            cv2.line(img, inb[k - 1], inb[k], (0, 0, 220), 2)
        # caption in white: never an object colour, per the recovery bug in
        # Code/event_utils.py
        cv2.putText(img, f"{ev['event']}  {'HIT' if hit else 'MISS'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)
        panels.append(img)

    rate = n_hit / len(events)
    print(f"\n[summary] {n_hit}/{len(events)} events hit  "
          f"(hit rate {rate:.3f})", flush=True)
    chance = float(np.mean([e["mask_frac"] for e in events]))
    print(f"[summary] mean mask coverage {chance:.3%} of frame, so a ray "
          f"placed at random would rarely hit", flush=True)
    if rate >= 0.8:
        print("[verdict] CONSISTENT with this transform. Not proof: "
              "confirmation still needs an independent source (Mark's CAD "
              "angle, or ChArUco hand-eye).", flush=True)
    else:
        print("[verdict] REJECTED at this transform.", flush=True)

    os.makedirs("figures", exist_ok=True)
    backup_if_exists(args.out_csv)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "event", "role", "img_id", "force_mag_N",
                    "hit", "dist_to_centroid_px", "ray_px_in_frame",
                    "mask_source", "mask_frac"])
        w.writerows(rows)
    print(f"[write] {args.out_csv}", flush=True)

    if panels:
        h = min(p.shape[0] for p in panels)
        res = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h))
               for p in panels]
        per_row = 3
        grid_rows = []
        for i in range(0, len(res), per_row):
            chunk = res[i:i + per_row]
            wmax = max(c.shape[1] for c in chunk)
            chunk = [cv2.copyMakeBorder(c, 0, 0, 0, wmax - c.shape[1],
                                        cv2.BORDER_CONSTANT, value=(20, 20, 20))
                     for c in chunk]
            while len(chunk) < per_row:
                chunk.append(np.full_like(chunk[0], 20))
            grid_rows.append(np.hstack(chunk))
        wmax = max(g.shape[1] for g in grid_rows)
        grid_rows = [cv2.copyMakeBorder(g, 0, 0, 0, wmax - g.shape[1],
                                        cv2.BORDER_CONSTANT, value=(20, 20, 20))
                     for g in grid_rows]
        backup_if_exists(args.out_png)
        cv2.imwrite(args.out_png, np.vstack(grid_rows))
        print(f"[write] {args.out_png}", flush=True)

    print("[note] calibration.yaml NOT modified, by design.", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
