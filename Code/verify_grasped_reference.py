"""Verify whether a recording's ``grasped`` reference track is a carried object.

Deliverable A's evaluation manifest may only admit a recording as grasped
ground truth if the reference track really follows an object carried by the
gripper. ``figures/t004`` and ``figures/t005`` ship propagated ``grasped``
tracks, but the manifest excluded them with an unverified note. This script
replaces that note with a measurement.

The test exploits the eye-in-hand geometry directly (the same algebraic
cancellation ``Code/grasped_seed_pixel.py`` relies on): the camera is rigidly
mounted to the moving hand, so

  * an object held by the gripper is fixed relative to the camera and projects
    to a near-constant pixel for the whole closed hold, regardless of how far
    the arm travels;
  * an object fixed in the world sweeps across the image exactly as fast as the
    camera moves.

So the discriminator is the reference mask centroid's drift across the hold,
measured against how much the scene itself moved over the same frames. The
background drift is required: if the arm barely moved, a stationary mask proves
nothing, and the recording is reported as inconclusive rather than carried.

No calibration and no extrinsic are involved. Scoring is per grasp cycle, and
the verdict is per cycle so a partially usable recording can contribute its
good cycles instead of being dropped whole.

SCOPE, measured 2026-09-03: this is supporting evidence, NOT the manifest's
admission gate. The test can only decide a cycle whose held phase overlaps real
camera motion, and on ``lfdws_t002_*`` it does not: the cube's reference mask
grows 30k -> 79k px across the closed run (the object is still approaching the
camera, so it is not yet rigidly held), and by the time the mask is stable the
arm has nearly stopped, leaving no usable segment. ``lfdws_t002`` therefore
reads ``not_carried`` despite a reference track that matches the object at IoU
0.986. Read a ``carried`` verdict as confirmation and anything else as "this
test could not decide", then fall back to the admission criterion recorded in
``config/evaluation_manifest.yaml``: the reference mask must be the object held
in the gripper at the early-hold frame the selector is actually scored on,
checked by inspecting that frame. A whole-hold ``not_carried`` verdict can
coexist with a valid early carried phase in a pick-and-place cycle.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deliverable_events import IMG, detect_events, find_demo_csv, load_demo_rows
from event_utils import mask_from_overlay

RGB_DIR = "zed_zed_node_rgb_color_rect_image_compressed"
GRASPED_COLOR = (0, 255, 0)

# A carried object may wobble in the fingers; it must still be far more static
# than the scene. These are structural thresholds, not per-recording tuning.
CARRIED_MAX_RATIO = 0.35           # object drift / background drift
MIN_BACKGROUND_DRIFT_FRACTION = 0.02  # below this the camera barely moved


def write_json_safely(path: str, payload: Dict) -> None:
    """Serialize completely before replacing the last good evidence file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2,
                  default=lambda value: value.item()
                  if isinstance(value, np.generic) else str(value))
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(path):
        shutil.copy2(path, path + ".bak")
    os.replace(tmp, path)


def probe_rows(cycle: Dict, count: int) -> List[int]:
    """Evenly spaced sample indices strictly inside the closed hold."""
    start, end = int(cycle["start_idx"]), int(cycle["end_idx"])
    span = max(1, end - start)
    fractions = np.linspace(0.2, 0.8, count)
    return sorted({max(start, min(end - 1, start + int(span * f)))
                   for f in fractions})


def centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def feature_drift(frame_paths: List[str], region: Optional[np.ndarray],
                  inside: bool, min_distance: int = 12,
                  max_steps: int = 60) -> Tuple[float, int]:
    """Accumulated image motion of tracked features, in pixels.

    Dense flow between two frames seconds apart is not usable here: the camera
    travels far enough that Farneback's median collapses toward zero and every
    recording looks static. Instead a grid of features is tracked frame to frame
    with Lucas-Kanade and chained across the hold, so the returned value is how
    far that content actually swept, first frame to last.

    ``inside=False`` tracks the scene with ``region`` masked out, so the
    candidate object cannot contribute its own motion to the reference it is
    compared against. ``inside=True`` tracks the object's own texture. Using one
    estimator for both sides is what makes the ratio meaningful: a propagated
    mask that grows as the object comes into view moves its centroid without the
    object moving at all, which is exactly the artefact that must not be read as
    world-fixed motion.
    """
    if len(frame_paths) < 2:
        return 0.0, 0
    step = max(1, len(frame_paths) // max_steps)
    paths = frame_paths[::step]
    if paths[-1] != frame_paths[-1]:
        paths.append(frame_paths[-1])

    previous = cv2.imread(paths[0])
    if previous is None:
        return 0.0, 0
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    track_mask = np.full(previous_gray.shape, 0 if inside else 255, dtype=np.uint8)
    if region is not None and region.shape == previous_gray.shape:
        track_mask[region] = 255 if inside else 0
    points = cv2.goodFeaturesToTrack(previous_gray, maxCorners=400,
                                     qualityLevel=0.01,
                                     minDistance=min_distance,
                                     mask=track_mask)
    if points is None or len(points) < 8:
        return 0.0, 0 if points is None else len(points)

    origin = points.copy()
    alive = np.ones(len(points), dtype=bool)
    for path in paths[1:]:
        frame = cv2.imread(path)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        moved, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, gray, points, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        if moved is None:
            break
        alive &= status.reshape(-1).astype(bool)
        points = moved
        previous_gray = gray
        if alive.sum() < 8:
            break
    if alive.sum() < 8:
        return 0.0, int(alive.sum())
    displacement = np.linalg.norm(
        points[alive].reshape(-1, 2) - origin[alive].reshape(-1, 2), axis=1)
    return float(np.median(displacement)), int(alive.sum())


def verdict(ratios: List[float]) -> Tuple[str, str]:
    """Classify from per-segment ratios that already passed their guards.

    Only segments where the camera demonstrably moved and the object was
    trackable reach here. An untrackable object must never arrive as a zero
    drift, which would read as a perfectly static, therefore carried, object.
    """
    if not ratios:
        return "inconclusive", "no_segment_with_camera_motion_and_trackable_object"
    ratio = float(np.median(ratios))
    if ratio <= CARRIED_MAX_RATIO:
        return "carried", "static_in_camera_while_scene_moved"
    return "not_carried", "object_tracks_scene_motion"


def evaluate_recording(recording: Dict, probes: int) -> List[Dict]:
    trial = recording["trial"]
    rgb_dir = os.path.join(trial, RGB_DIR)
    rows = load_demo_rows(find_demo_csv(trial))
    _, cycles, summary = detect_events(rows)
    reference = list(csv.DictReader(open(recording["reference_sidecar"])))
    by_frame = {r["img_filename"]: r for r in reference
                if r.get("role") == "grasped"}

    out = []
    for cycle in cycles:
        if cycle.get("grasp") is None:
            continue
        result = {"recording": recording["id"], "cycle_idx": cycle["cycle_idx"],
                  "hold_rows": [int(cycle["start_idx"]), int(cycle["end_idx"])]}
        samples = []
        for row_idx in probe_rows(cycle, probes):
            img_id = str(rows[row_idx][IMG])
            reference_row = by_frame.get(f"{img_id}.png")
            image = cv2.imread(os.path.join(rgb_dir, f"{img_id}.png"))
            if reference_row is None or image is None:
                continue
            mask = mask_from_overlay(reference_row["overlay_path"],
                                     os.path.join(rgb_dir, f"{img_id}.png"),
                                     GRASPED_COLOR)
            if mask is None or not mask.any():
                continue
            point = centroid(mask)
            if point is None:
                continue
            samples.append({"row_idx": row_idx, "img_id": img_id,
                            "centroid": point, "mask_px": int(mask.sum()),
                            "image": image, "mask": mask})
            print(f"[probe] {recording['id']} c{cycle['cycle_idx']} {img_id} "
                  f"centroid=({point[0]:.1f},{point[1]:.1f}) "
                  f"px={int(mask.sum())}", flush=True)

        if len(samples) < 2:
            result.update(verdict="inconclusive", reason="too_few_reference_frames",
                          probe_count=len(samples))
            out.append(result)
            continue

        # Score segment by segment, not first frame to last. A single
        # whole-hold ratio is dominated by whichever stretch happens to be
        # longest, and on these recordings the arm is often nearly stationary
        # for part of the hold, where the ratio is a small number divided by a
        # smaller one and means nothing.
        h, w = samples[0]["image"].shape[:2]
        diagonal = float(np.hypot(w, h))
        segments, ratios = [], []
        for first, second in zip(samples, samples[1:]):
            paths = [os.path.join(rgb_dir, f"{rows[i][IMG]}.png")
                     for i in range(first["row_idx"], second["row_idx"] + 1)]
            scene_drift, scene_points = feature_drift(paths, first["mask"],
                                                      inside=False)
            object_drift, object_points = feature_drift(
                paths, first["mask"], inside=True, min_distance=4)
            usable = (scene_points >= 8 and object_points >= 8 and
                      scene_drift >= MIN_BACKGROUND_DRIFT_FRACTION * diagonal)
            ratio = object_drift / max(scene_drift, 1e-6)
            segment = {"rows": [first["row_idx"], second["row_idx"]],
                       "mask_px": first["mask_px"],
                       "object_drift_px": round(object_drift, 2),
                       "object_points": object_points,
                       "background_drift_px": round(scene_drift, 2),
                       "background_points": scene_points,
                       "ratio": round(ratio, 3), "usable": bool(usable)}
            segments.append(segment)
            if usable:
                ratios.append(ratio)
            print(f"[segment] {recording['id']} c{cycle['cycle_idx']} "
                  f"{first['row_idx']}-{second['row_idx']} "
                  f"obj={object_drift:.1f}(n={object_points}) "
                  f"bg={scene_drift:.1f}(n={scene_points}) "
                  f"ratio={ratio:.2f} usable={usable}", flush=True)

        decision, reason = verdict(ratios)
        mask_drift = float(np.median(ratios)) if ratios else float("nan")
        result.update(verdict=decision, reason=reason, probe_count=len(samples),
                      median_usable_ratio=None if not ratios else round(mask_drift, 3),
                      usable_segments=len(ratios), segments=segments,
                      image_diagonal_px=round(diagonal, 1),
                      mask_px=[s["mask_px"] for s in samples],
                      frames=[s["img_id"] for s in samples])
        ratio_text = "n/a" if not ratios else f"{mask_drift:.2f}"
        print(f"[verdict] {recording['id']} c{cycle['cycle_idx']}: {decision} "
              f"({reason}) median_ratio={ratio_text} "
              f"usable_segments={len(ratios)}/{len(segments)}", flush=True)
        out.append(result)

    if not out:
        print(f"[skip] {recording['id']}: no grasp cycles "
              f"(has_gripper={summary['has_gripper']})", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="config/evaluation_manifest.yaml")
    ap.add_argument("--recording", action="append", default=[],
                    help="recording id to check; default is every recording")
    ap.add_argument("--probes", type=int, default=5)
    ap.add_argument("--out", default="figures/grasped_reference_verification.json")
    args = ap.parse_args()

    manifest = yaml.safe_load(open(args.manifest))
    wanted = set(args.recording)
    results = []
    for recording in manifest["recordings"]:
        if wanted and recording["id"] not in wanted:
            continue
        if not os.path.isdir(recording["trial"]):
            print(f"[skip] {recording['id']}: trial directory missing", flush=True)
            continue
        if not os.path.exists(recording["reference_sidecar"]):
            print(f"[skip] {recording['id']}: reference sidecar missing", flush=True)
            continue
        results.extend(evaluate_recording(recording, args.probes))

    write_json_safely(args.out, {"thresholds": {
        "carried_max_ratio": CARRIED_MAX_RATIO,
        "min_background_drift_fraction": MIN_BACKGROUND_DRIFT_FRACTION},
        "cycles": results})
    print(f"[write] {args.out} cycles={len(results)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
