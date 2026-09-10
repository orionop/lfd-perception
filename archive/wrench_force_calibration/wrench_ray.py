"""
Shared geometry for wrench-line projection into the eye-in-hand camera.

Pulled out so that the extrinsic search (Code/extrinsic_grid_search.py) and
the standalone validator (Code/wrench_ray_validate.py) evaluate exactly the
same math. project_ee.py keeps its own copy for now and is untouched.

THE GEOMETRY
------------
current_pose publishes the pose of the BOTA SENSONE ORIGIN in the base frame,
so the camera pose is a per-frame quantity:

    T_base_camera(t) = T_base_bota(t) @ T_bota_camera

with T_bota_camera the one fixed unknown. The wrench is already expressed in
the bota frame, so no extra mount transform is needed.

From Bicchi 1990, the line of action of a pure force measured as (f, tau) at
the sensor origin is

    r(s) = r0 + s * fhat ,   r0 = (f x tau)/|f|^2 ,   fhat = f/|f|

r0 is the point on that line closest to the sensor origin and is orthogonal
to fhat by construction, which the caller can assert as a cheap check. The
physical contact point lies somewhere along the line, which is why this gives
a ray to search rather than a single point.

THE PARAMETERISATION
--------------------
Mark's CAD (2026-08-18, rev2) fixes the gripper centre to camera translation
as (0, 158.82, 91.72) mm, magnitude 183.41 mm at exactly 30.007 degrees, with
the third component confirmed zero to within 8 microns. It does NOT fix the
orientation, which the bracket tilt sets independently.

What stays free, and the bound on each:

  psi    azimuth of the camera offset about the bota Z axis. Free because the
         CAD calls the dimension "dY" but we have not confirmed the CAD Y
         axis is the bota Y axis.
  tilt   pitch of the optical axis below horizontal. This is the one that
         matters, since rotation error scales with target range while
         translation error does not.
  d      bota origin to gripper centre along the tool axis, bounded to
         104 to 142 mm: image 2 gives 104 mm from a mounting plane to the
         gripper centre, and the SensONE body is 38.00 mm thick, so the
         sensor origin sits somewhere inside that span.
  lens   lateral shift along camera X. The ZED Mini baseline is 63 mm and the
         rectified RGB stream is the LEFT camera, so the candidate set covers
         both "Coordinate System1 is at a lens" and "it is at the body
         centre".

Roll about the optical axis is assumed zero, that is, the stereo baseline
stays level. That is what this bracket is designed to do and it matches the
existing calibration.yaml finding that the baseline direction carries no tilt
component. It is an assumption, not a measurement.
"""
import numpy as np

# Mark's CAD dimensions, millimetres. Verified internally consistent:
# sqrt(158.82^2 + 91.72^2) = 183.402 against the stated 183.41.
CAD_OFFSET_LATERAL_MM = 158.82
CAD_OFFSET_UP_MM = 91.72
CAD_DIST_MM = 183.41

# bota origin -> gripper centre along the tool axis, millimetres. Lower bound
# is image 2's 104 mm; upper bound adds the SensONE's 38.00 mm body thickness.
D_AXIAL_MIN_MM = 104.0
D_AXIAL_MAX_MM = 142.0

# ZED Mini stereo baseline, Stereolabs datasheet.
ZED_BASELINE_MM = 63.0

# Candidate lateral shifts of the optical centre along camera X.
# +/-63 covers "Coordinate System1 sits at the other lens", +/-31.5 covers
# "it sits at the body centre", 0 covers "it is already the left lens".
LENS_SHIFT_CANDIDATES_MM = (-63.0, -31.5, 0.0, 31.5, 63.0)


def quat_to_R(x, y, z, w):
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def pose_to_T(px, py, pz, qx, qy, qz, qw):
    """One current_pose row -> 4x4 T_base_bota(t)."""
    T = np.eye(4)
    T[:3, :3] = quat_to_R(qx, qy, qz, qw)
    T[:3, 3] = [px, py, pz]
    return T


def wrench_line_bota(f, tau):
    """Bicchi 1990 line of action, in the bota frame.

    Returns (r0, fhat), or (None, None) when the force is too small for the
    line to be defined.
    """
    f = np.asarray(f, dtype=float)
    tau = np.asarray(tau, dtype=float)
    fn2 = float(f @ f)
    if fn2 < 1e-9:
        return None, None
    r0 = np.cross(f, tau) / fn2
    fhat = f / np.sqrt(fn2)
    return r0, fhat


def make_T_bota_camera(psi_deg, tilt_deg, d_axial_mm, lens_shift_mm,
                       lateral_mm=CAD_OFFSET_LATERAL_MM,
                       up_mm=CAD_OFFSET_UP_MM):
    """Candidate T_bota_camera, mapping a camera-frame point to the bota frame.

    psi_deg        azimuth of the camera offset about the bota Z axis
    tilt_deg       pitch of the optical axis below horizontal; 0 looks
                   horizontally back along the offset direction, 90 looks
                   straight down the tool axis
    d_axial_mm     bota origin -> gripper centre along the tool axis
    lens_shift_mm  lateral shift of the optical centre along camera X
    """
    psi = np.radians(float(psi_deg))
    th = np.radians(float(tilt_deg))

    u = np.array([np.cos(psi), np.sin(psi), 0.0])
    zb = np.array([0.0, 0.0, 1.0])

    # Optical centre in the bota frame, metres. The gripper centre sits d
    # below the bota origin along the tool axis; the camera is then `lateral`
    # out along u and `up` above the gripper centre.
    p = u * (lateral_mm * 1e-3) + zb * ((up_mm - float(d_axial_mm)) * 1e-3)

    # Optical axis, tilted down from horizontal and pointing back across the
    # gripper so the camera sees the workspace past the fingertips.
    z_cam = -np.cos(th) * u - np.sin(th) * zb
    z_cam = z_cam / np.linalg.norm(z_cam)

    # Camera X kept horizontal so the stereo baseline stays level.
    x_cam = np.cross(zb, u)
    nx = np.linalg.norm(x_cam)
    x_cam = np.array([1.0, 0.0, 0.0]) if nx < 1e-9 else x_cam / nx
    y_cam = np.cross(z_cam, x_cam)
    y_cam = y_cam / np.linalg.norm(y_cam)

    p = p + x_cam * (float(lens_shift_mm) * 1e-3)

    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = x_cam, y_cam, z_cam
    T[:3, 3] = p
    return T


def project_points(pts_base, K, T_base_cam):
    """(N,3) base-frame points -> (N,2) pixels plus (N,) camera-frame depth.

    Pure pinhole, no distortion: the exported stream is already rectified and
    calibration.yaml carries zero distortion coefficients for it.
    """
    T_cam_base = np.linalg.inv(np.asarray(T_base_cam, dtype=float))
    P = np.hstack([np.asarray(pts_base, dtype=float),
                   np.ones((len(pts_base), 1))])
    cam = (T_cam_base @ P.T).T[:, :3]
    z = cam[:, 2]
    safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
    K = np.asarray(K, dtype=float)
    u = K[0, 0] * cam[:, 0] / safe + K[0, 2]
    v = K[1, 1] * cam[:, 1] / safe + K[1, 2]
    return np.stack([u, v], axis=1), z


def ray_pixels(r0_bota, fhat_bota, T_base_bota, T_bota_camera, K,
               s_min=-0.05, s_max=0.60, n=140):
    """Sample the wrench line and project it, keeping only points in front of
    the camera so a ray pointing away yields nothing rather than a mirrored
    ghost."""
    s = np.linspace(s_min, s_max, n)
    pts_bota = r0_bota[None, :] + s[:, None] * fhat_bota[None, :]
    P = np.hstack([pts_bota, np.ones((len(pts_bota), 1))])
    T_base_bota = np.asarray(T_base_bota, dtype=float)
    pts_base = (T_base_bota @ P.T).T[:, :3]
    T_base_cam = T_base_bota @ np.asarray(T_bota_camera, dtype=float)
    uv, z = project_points(pts_base, K, T_base_cam)
    keep = z > 1e-3
    return uv[keep], z[keep]


def ray_mask_score(uv, mask, img_w, img_h):
    """Agreement between a projected ray and a ground truth mask.

    Returns (hit, dist_px, n_in_image):
      hit         any sampled ray pixel falls inside the mask
      dist_px     distance from the closest in-image ray pixel to the mask
                  centroid, inf if the ray never enters the image
      n_in_image  how many sampled points landed in frame at all

    Distance to the centroid rather than to the mask boundary keeps the score
    smooth, which matters because the search is a coarse grid.
    """
    if len(uv) == 0:
        return False, float("inf"), 0
    u, v = uv[:, 0], uv[:, 1]
    inb = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    n_in = int(inb.sum())
    if n_in == 0:
        return False, float("inf"), 0
    ui = np.clip(u[inb].astype(int), 0, img_w - 1)
    vi = np.clip(v[inb].astype(int), 0, img_h - 1)
    hit = bool(mask[vi, ui].any())
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return hit, float("inf"), n_in
    cx, cy = xs.mean(), ys.mean()
    d = float(np.min(np.hypot(u[inb] - cx, v[inb] - cy)))
    return hit, d, n_in
