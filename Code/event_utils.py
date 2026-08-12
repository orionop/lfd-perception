"""
Shared, corrected primitives for event detection and overlay-mask recovery.

Created 2026-08-12 after an audit found two real bugs that were duplicated
across ~12 scripts because each had its own copy of the same logic. New
code should import from here rather than re-implementing either function.

--------------------------------------------------------------------------
BUG 1 -- phantom grasp/release from gripper sensor noise
--------------------------------------------------------------------------
Every copy of the gripper detector computed its open/closed threshold as
the midpoint of the observed width range:

    thr = w_closed + 0.5 * (w_open - w_closed)

with no check that the gripper actually MOVED. On a recording where the
gripper never actuates, w_open and w_closed collapse onto the sensor's
own noise floor, the midpoint lands inside that noise band, and the
detector manufactures grasp/release events out of nothing.

Confirmed on lfdws_t001_labexport: gripper width spans
0.07999710 -> 0.07999776 m, i.e. 6.6e-7 m (0.66 micrometres) of pure
noise, yet the detector reported grasp at t=0.06s and release at t=7.66s.

This is not cosmetic. Because the press/contact search is deliberately
restricted to the window between grasp and release, a spurious window
also corrupts the contact event: that trial's sidecar reported press at
t=3.34s when the true force peak is at t=5.08s (|F|=11.15N).

Fix: require a minimum travel of MIN_GRIPPER_SPAN_M before believing any
gripper transition. Real grasps in this dataset span 0.024-0.080 m, so a
1 mm floor sits ~24x below the smallest real motion and ~1500x above the
observed noise -- a wide, safe margin. When the guard trips, the detector
degrades to the force-only path, which is the behaviour that was always
intended for a trial with no real gripper cycle.

--------------------------------------------------------------------------
BUG 2 -- overlay caption recovered as object mask
--------------------------------------------------------------------------
The sidecar builders reconstruct each object's mask from its propagation
overlay PNG by colour-differencing against the source frame. The
propagation scripts (propagate_cup.py, propagate_demo_bidir.py,
propagate_object_n.py) draw their per-frame caption in the SAME BGR
colour as that object's mask, so the caption glyphs were recovered as
object pixels -- producing a phantom ~700-1050 px "object" with a fixed
bbox at (10,16)-(159,35) on frames where the object is genuinely absent,
and silently inflating mask_px on every frame where it is present.

The discriminator is opacity, not colour: masks are alpha-blended
(cv2.addWeighted(img, 1.0, layer, 0.5, 0)), which leaves the colour's
"off" channels UNCHANGED, whereas cv2.putText writes the colour solid,
which forces the off channels to 0. The original test accepted any
off-channel value below +tol, so both passed. Requiring the off channels
to be genuinely unchanged (|diff| < tol) rejects solid text while keeping
blended mask pixels.

Measured on lfdws_t001_depth's plate track: phantom frames drop
1012 -> 26 px, 959 -> 24 px, 1004 -> 24 px (removed entirely once the
MIN_MASK_PX floor is applied), while real masks are preserved to within
~2% (81343 -> 79973, 89862 -> 88517) -- the small loss being
anti-aliased edge pixels, which the stricter test is correct to exclude.

Note this recovery is inherently lossy on bright objects: a mask pixel
whose source channel is already near 255 saturates under the 0.5 blend,
so its diff falls below tol and it is dropped. Reported mask_px is
therefore a lower bound on true mask area. The authoritative per-frame
count remains the propagation summary CSV, which records mask.sum()
directly. Downstream code needing exact areas should prefer that.
"""
import ast

import cv2
import numpy as np

# Minimum gripper travel (metres) before a width transition is believed.
# Real grasps here span 0.024-0.080 m; observed no-motion noise is 6.6e-7 m.
MIN_GRIPPER_SPAN_M = 0.001

# Recovered masks smaller than this are treated as empty (anti-aliasing
# residue from the rejected caption, typically <30 px).
MIN_MASK_PX = 50

# Height (px) of the caption band drawn at the top-left of every overlay.
# Captions are anchored at y=30 with ~0.6 scale, so glyphs occupy roughly
# y in [14, 34]; 40 covers it with margin.
CAPTION_BAND_H = 40


def parse_gripper_width(cell):
    """ros2_unbag writes JointState.position as one bracketed string."""
    try:
        return float(np.sum(ast.literal_eval(cell)))
    except Exception:
        return float("nan")


def gripper_moved(widths):
    """True if the gripper's travel exceeds the sensor noise floor.

    `widths` is the per-sample summed finger width. Returns False when the
    gripper never actuated, in which case grasp/release must NOT be
    inferred (see BUG 1 above).
    """
    w = np.asarray(widths, dtype=float)
    if not np.isfinite(w).any():
        return False
    span = float(np.nanmax(w) - np.nanmin(w))
    return span >= MIN_GRIPPER_SPAN_M


def gripper_transitions(widths):
    """Return (grasp_idx, release_idx), either possibly None.

    Applies the minimum-span guard, so a gripper that never moved yields
    (None, None) rather than transitions manufactured from noise.
    """
    w = np.asarray(widths, dtype=float)
    if not gripper_moved(w):
        return None, None
    w_open, w_closed = float(np.nanmax(w)), float(np.nanmin(w))
    thr = w_closed + 0.5 * (w_open - w_closed)
    closed = w < thr
    cd = np.where((~closed[:-1]) & (closed[1:]))[0] + 1
    cu = np.where((closed[:-1]) & (~closed[1:]))[0] + 1
    return (int(cd[0]) if len(cd) else None,
            int(cu[-1]) if len(cu) else None)


def gripper_closed_window(widths):
    """Boolean mask of 'gripper is closed' samples, all-False if it never
    moved (so a caller restricting a force search to this window falls
    back to searching the whole recording rather than a phantom one)."""
    w = np.asarray(widths, dtype=float)
    if not gripper_moved(w):
        return np.zeros(len(w), dtype=bool)
    w_open, w_closed = float(np.nanmax(w)), float(np.nanmin(w))
    return w < w_closed + 0.5 * (w_open - w_closed)


def mask_from_overlay(ov_path, src_path, color_bgr, tol=40,
                      min_px=MIN_MASK_PX):
    """Recover an alpha-blended mask from an overlay PNG, rejecting the
    solid-colour caption drawn in the same colour (see BUG 2 above).

    Returns a boolean mask, or None if either image is unreadable or the
    shapes disagree.
    """
    ov = cv2.imread(ov_path)
    src = cv2.imread(src_path)
    if ov is None or src is None or ov.shape != src.shape:
        return None
    diff = ov.astype(int) - src.astype(int)
    b, g, r = color_bgr
    m = np.ones(diff.shape[:2], dtype=bool)
    for ch, target in zip(range(3), (b, g, r)):
        if target > 100:
            # channel the colour drives up: blending raises it
            m &= diff[..., ch] > tol
        else:
            # channel the colour leaves alone: blending must NOT change it.
            # Solid caption text forces this to 0 (large negative diff) and
            # is rejected here; the old `diff < tol` accepted it.
            m &= np.abs(diff[..., ch]) < tol
    # Overlays generated BEFORE the caption colour was changed to white
    # still carry a caption in the object's own colour. The off-channel
    # test above rejects most of it, but glyph pixels survive wherever the
    # source frame is already ~0 in that channel (dark background), so the
    # off-channel diff is ~0 and looks "unchanged". Such survivors are
    # confined to the caption band, so drop any connected component that
    # lies entirely within it -- a real object overlapping the top of the
    # frame extends below the band and is kept intact.
    if m.any():
        n_lab, labels = cv2.connectedComponents(m.astype(np.uint8))
        for lab in range(1, n_lab):
            comp = labels == lab
            if not comp[CAPTION_BAND_H:, :].any():
                m &= ~comp
    if m.sum() < min_px:
        return np.zeros(diff.shape[:2], dtype=bool)
    return m
