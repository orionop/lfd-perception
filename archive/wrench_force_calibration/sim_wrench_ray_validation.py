"""
Analytical pre-flight sanity check for the Bicchi wrench-ray recovery math
used in Code/project_ee.py -- NOT a substitute for real-hardware
validation, and not presented as one. See Docs/publication.tex's
Limitations section and the plan this was built from: the paper's central
claim is that this method overcomes a real-world calibration difficulty
(two failed CAD-based bota_to_camera attempts, documented in
Code/cad_candidate_sensitivity.py). A simulation that DEFINES the camera
extrinsic exactly sidesteps that exact difficulty, so it cannot serve as
evidence for the claim -- it can only answer a narrower, legitimate
question: given the sensor's REAL published noise floor, is the
closed-form recovery r0 = (f x tau)/|f|^2 numerically well-conditioned,
or does noise blow it up? That's a pre-flight check on the math, done
before spending real rig time, not a paper result.

METHOD: forward-simulate a single frictionless point contact. Bicchi's
r0 is defined as the point on the force's line of action closest to the
sensor origin, which by construction is perpendicular to the force
direction (r0 . f_hat = 0) -- this is what makes the closed-form
recovery exact in the noise-free case (verified below before any noise
is added, same "validate the trivial case first" discipline used in
Code/calibrate_hand_eye.py). Given a ground-truth (r0_true, f_hat_true,
|F|), the induced torque about the sensor origin is tau_true = r0_true x
f_true (pure moment from a point force, no additional applied couple --
the explicit single-contact assumption this whole method relies on).
Gaussian sensor noise is added at Bota SensONE's own published
noise-free-resolution spec (peak-to-peak 6-sigma, converted to per-axis
Gaussian std = spec/6): 0.3 N on Fx/Fy/Fz, 0.007 Nm on Mx/My, 0.0025 Nm
on Mz (verified from the vendor datasheet, not assumed). The SAME
recovery code path used in Code/project_ee.py is then run on the noisy
(f, tau) and the recovered r0/f_hat are compared to ground truth, both
in 3D (mm) and, projected through the REAL trusted camera intrinsics
from calibration.yaml, in image pixels.

Swept over: contact-force magnitude (5-27 N, the range actually observed
across this repo's real trials) x contact geometry (random r0 direction
and offset each draw) x noise (real sensor spec). n=500 draws per force
level, seeded for reproducibility.

Output: figures/sim_wrench_ray_sanity_check.png (error vs. force
magnitude, both 3D and pixel error) + printed summary table.

Run inside .venv_analysis (only needs numpy/opencv/matplotlib):
    .venv_analysis/bin/python Code/sim_wrench_ray_validation.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))
from project_ee import pose_to_T, project_points, quat_to_R  # reuse, don't duplicate

# Bota SensONE 1.3 datasheet: noise-free resolution = peak-to-peak (6-sigma)
# noise with no load, 100 Hz. Convert to per-axis Gaussian std.
NOISE_FREE_RES_F = 0.3     # N, Fx/Fy/Fz
NOISE_FREE_RES_MXY = 0.007  # Nm, Mx/My
NOISE_FREE_RES_MZ = 0.0025  # Nm, Mz
STD_F = NOISE_FREE_RES_F / 6.0
STD_MXY = NOISE_FREE_RES_MXY / 6.0
STD_MZ = NOISE_FREE_RES_MZ / 6.0

FORCE_LEVELS_N = [5.0, 10.0, 15.0, 20.0, 27.0]  # observed range across real trials
N_DRAWS = 500
SEED = 0

OUT_PNG = "figures/sim_wrench_ray_sanity_check.png"


def random_perp_unit(f_hat, rng):
    """Random unit vector perpendicular to f_hat (uniform over the plane)."""
    v = rng.normal(size=3)
    v = v - (v @ f_hat) * f_hat
    return v / np.linalg.norm(v)


def zero_noise_check():
    """Verify exact recovery with zero noise BEFORE trusting anything
    downstream -- same discipline as Code/calibrate_hand_eye.py's
    synthetic pre-test."""
    rng = np.random.default_rng(SEED)
    f_hat_true = np.array([0.1, 0.15, -0.98])
    f_hat_true /= np.linalg.norm(f_hat_true)
    r0_dir = random_perp_unit(f_hat_true, rng)
    r0_true = r0_dir * 0.04  # 4cm offset, plausible given real project_ee.py prints
    f_true = 15.0 * f_hat_true
    tau_true = np.cross(r0_true, f_true)

    fmag = np.linalg.norm(f_true)
    f_hat_rec = f_true / fmag
    r0_rec = np.cross(f_true, tau_true) / fmag**2

    pos_err = np.linalg.norm(r0_rec - r0_true)
    ang_err = np.degrees(np.arccos(np.clip(f_hat_rec @ f_hat_true, -1, 1)))
    print(f"[zero-noise check] position error: {pos_err:.2e} m  "
          f"direction error: {ang_err:.2e} deg", flush=True)
    assert pos_err < 1e-9 and ang_err < 1e-6, "zero-noise recovery is not exact -- bug"
    print("[zero-noise check] PASS -- closed-form recovery is exact when noise-free",
          flush=True)


def main():
    zero_noise_check()

    with open("calibration.yaml") as f:
        calib = yaml.safe_load(f)
    K = calib["camera_intrinsics"]["K"]
    dist = calib["camera_intrinsics"]["dist"]

    # arbitrary but fixed scene for pixel-error reporting -- NOT the real
    # rig's (unknown) transform, just a plausible eye-in-hand-like pose
    # so pixel error has a concrete, reportable scale
    T_base_bota = pose_to_T(0.4, 0.1, 0.35, 0.0, 0.0, 0.0, 1.0)
    T_bota_camera = np.eye(4)
    T_bota_camera[:3, :3] = np.array([
        [-0.7071, -0.5, -0.5], [-0.7071, 0.5, 0.5], [0.0, 0.7071, -0.7071]])
    T_bota_camera[:3, 3] = [0.11, -0.14, 0.03]
    T_base_cam = T_base_bota @ T_bota_camera

    rng = np.random.default_rng(SEED)
    results = {}
    for force_n in FORCE_LEVELS_N:
        pos_errs_mm, ang_errs_deg, px_errs = [], [], []
        for _ in range(N_DRAWS):
            f_hat_true = rng.normal(size=3)
            f_hat_true /= np.linalg.norm(f_hat_true)
            r0_dir = random_perp_unit(f_hat_true, rng)
            r0_offset_m = rng.uniform(0.01, 0.08)  # 1-8cm, plausible contact offsets
            r0_true = r0_dir * r0_offset_m
            f_true = force_n * f_hat_true
            tau_true = np.cross(r0_true, f_true)

            f_noisy = f_true + rng.normal(0, STD_F, size=3)
            tau_noisy = tau_true + rng.normal(0, [STD_MXY, STD_MXY, STD_MZ], size=3)

            fmag = np.linalg.norm(f_noisy)
            if fmag < 1e-6:
                continue
            f_hat_rec = f_noisy / fmag
            r0_rec = np.cross(f_noisy, tau_noisy) / fmag**2

            pos_errs_mm.append(np.linalg.norm(r0_rec - r0_true) * 1000)
            ang_errs_deg.append(np.degrees(
                np.arccos(np.clip(f_hat_rec @ f_hat_true, -1, 1))))

            r0_true_base = T_base_bota[:3, :3] @ r0_true + T_base_bota[:3, 3]
            r0_rec_base = T_base_bota[:3, :3] @ r0_rec + T_base_bota[:3, 3]
            uv, _ = project_points(np.vstack([r0_true_base, r0_rec_base]),
                                   K, T_base_cam, dist)
            px_errs.append(float(np.linalg.norm(uv[0] - uv[1])))

        results[force_n] = {
            "pos_mm": np.array(pos_errs_mm),
            "ang_deg": np.array(ang_errs_deg),
            "px": np.array(px_errs),
        }
        print(f"[F={force_n:5.1f}N] n={len(pos_errs_mm)}  "
              f"pos_err: mean={np.mean(pos_errs_mm):.2f}mm median={np.median(pos_errs_mm):.2f}mm "
              f"p95={np.percentile(pos_errs_mm,95):.2f}mm  "
              f"dir_err: mean={np.mean(ang_errs_deg):.2f}deg  "
              f"pixel_err: mean={np.mean(px_errs):.1f}px median={np.median(px_errs):.1f}px",
              flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    levels = FORCE_LEVELS_N
    axes[0].boxplot([results[f]["pos_mm"] for f in levels], tick_labels=[f"{f:.0f}" for f in levels])
    axes[0].set_xlabel("contact force (N)"); axes[0].set_ylabel("position error (mm)")
    axes[0].set_title("3D contact-point recovery error")
    axes[1].boxplot([results[f]["ang_deg"] for f in levels], tick_labels=[f"{f:.0f}" for f in levels])
    axes[1].set_xlabel("contact force (N)"); axes[1].set_ylabel("direction error (deg)")
    axes[1].set_title("Force-direction recovery error")
    axes[2].boxplot([results[f]["px"] for f in levels], tick_labels=[f"{f:.0f}" for f in levels])
    axes[2].set_xlabel("contact force (N)"); axes[2].set_ylabel("pixel error (px)")
    axes[2].set_title("Projected pixel error\n(arbitrary fixed scene, illustrative scale only)")
    fig.suptitle("Pre-flight sanity check ONLY -- Bota SensONE's real noise spec, "
                 "NOT a substitute for real-hardware validation", fontsize=10)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    print(f"\n[write] {OUT_PNG}", flush=True)
    print("[done] pre-flight sanity check only -- does NOT validate the real "
          "rig's camera calibration, which remains the open problem", flush=True)


if __name__ == "__main__":
    main()
