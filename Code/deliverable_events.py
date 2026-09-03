"""Canonical event detection for Deliverable A.

All production entry points should use this module instead of copying the
gripper midpoint and force-peak logic. It handles multiple grasp cycles and
gracefully degrades when either sensor stream is missing.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

from event_utils import closed_runs, parse_gripper_width

POSE_TS = "NS_1.franka_robot_state_broadcaster.current_pose.timestamp"
GRIP = "NS_1.franka_gripper.joint_states.position"
FX = "bota_post.wrench_body_compensated.wrench.force.x"
FY = "bota_post.wrench_body_compensated.wrench.force.y"
FZ = "bota_post.wrench_body_compensated.wrench.force.z"
IMG = "zed.zed_node.rgb.color.rect.image.compressed"


def find_demo_csv(trial: str) -> str:
    candidates = sorted(
        f for f in os.listdir(trial) if f.endswith(".csv") and not f.startswith(".")
    )
    preferred = [f for f in candidates if f.endswith("_0.csv")]
    if not candidates:
        raise FileNotFoundError(f"no merged CSV in {trial}")
    return os.path.join(trial, (preferred or candidates)[0])


def load_demo_rows(csv_path: str) -> List[Dict]:
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def _relative_seconds(values) -> np.ndarray:
    try:
        raw = np.array([float(v) for v in values], dtype=float)
        delta = raw - raw[0]
        # Exporters use integer nanoseconds; tolerate an already-second timeline.
        scale = 1e9 if np.nanmedian(np.abs(raw)) > 1e12 else 1.0
        return delta / scale
    except (TypeError, ValueError):
        parsed = [datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                  for v in values]
        return np.array([(value - parsed[0]).total_seconds() for value in parsed])


def _separated_peaks(signal: np.ndarray, threshold: float, distance: int) -> np.ndarray:
    """Small dependency-free equivalent of the find_peaks behavior we need."""
    finite = np.isfinite(signal)
    local = np.zeros(len(signal), dtype=bool)
    if len(signal) >= 3:
        local[1:-1] = (signal[1:-1] >= signal[:-2]) & (signal[1:-1] > signal[2:])
    candidates = np.flatnonzero(local & finite & (signal >= threshold))
    chosen = []
    for idx in sorted(candidates, key=lambda i: float(signal[i]), reverse=True):
        if all(abs(int(idx) - kept) >= distance for kept in chosen):
            chosen.append(int(idx))
    return np.array(sorted(chosen), dtype=int)


def _event(idx: int, name: str, t_rel, img_ids, force_mag, widths) -> Dict:
    return {
        "event": name,
        "row_idx": int(idx),
        "t_rel_s": float(t_rel[idx]),
        "img_ts": str(img_ids[idx]),
        "force_mag_n": None if force_mag is None else float(force_mag[idx]),
        "gripper_width_m": None if widths is None else float(widths[idx]),
    }


def detect_events(rows: List[Dict], force_peak_min_n: float = 5.0,
                  peak_distance_s: float = 0.5,
                  force_cluster_gap_s: float = 3.0) -> Tuple[List[Dict], List[Dict], Dict]:
    """Return ``(events, cycles, summary)`` for merged CSV dictionary rows."""
    if not rows:
        raise ValueError("merged CSV is empty")
    columns = rows[0]
    missing = [c for c in (POSE_TS, IMG) if c not in columns]
    if missing:
        raise ValueError(f"merged CSV missing required columns: {missing}")

    t_rel = _relative_seconds([row[POSE_TS] for row in rows])
    img_ids = np.array([str(row[IMG]) for row in rows])
    has_gripper = GRIP in columns
    has_force = all(c in columns for c in (FX, FY, FZ))

    widths = None
    runs: List[Tuple[int, int]] = []
    if has_gripper:
        widths = np.array([parse_gripper_width(row[GRIP]) for row in rows], dtype=float)
        runs = closed_runs(widths)

    force_mag = baseline = adjusted = None
    peaks = np.array([], dtype=int)
    if has_force:
        force_mag = np.sqrt(sum(np.array([float(row[c]) for row in rows]) ** 2
                                for c in (FX, FY, FZ)))
        n0 = max(1, len(force_mag) // 10)
        baseline = float(np.nanmedian(force_mag[:n0]))
        adjusted = force_mag - baseline
        dt = float(np.nanmedian(np.diff(t_rel))) if len(t_rel) > 1 else 0.01
        distance = max(1, int(round(peak_distance_s / max(dt, 1e-6))))
        eligible = np.zeros(len(rows), dtype=bool)
        if runs:
            for start, end in runs:
                eligible[start:end] = True
        else:
            eligible[:] = True
        signal = np.where(eligible, adjusted, -np.inf)
        peaks = _separated_peaks(signal, force_peak_min_n, distance)

    cycles: List[Dict] = []
    events: List[Dict] = []
    for cycle_idx, (start, end) in enumerate(runs, 1):
        release_idx = min(end, len(rows) - 1)
        presses = [int(p) for p in peaks if start <= p < end]
        grasp = _event(start, "grasp", t_rel, img_ids, force_mag, widths)
        release = _event(release_idx, "release", t_rel, img_ids,
                         force_mag, widths)
        packed_presses = [_event(p, "press", t_rel, img_ids,
                                 force_mag, widths) for p in presses]
        cycle = {"cycle_idx": cycle_idx, "start_idx": start, "end_idx": end,
                 "grasp": grasp, "release": release, "presses": packed_presses}
        cycles.append(cycle)
        events.extend([grasp, *packed_presses, release])

    representative_peaks = list(map(int, peaks))
    if not runs and len(peaks):
        clusters = []
        current = []
        for peak in map(int, peaks):
            if current and t_rel[peak] - t_rel[current[-1]] > force_cluster_gap_s:
                clusters.append(current)
                current = []
            current.append(peak)
        if current:
            clusters.append(current)
        representative_peaks = [max(cluster, key=lambda i: force_mag[i])
                                for cluster in clusters]
        for cycle_idx, p in enumerate(representative_peaks, 1):
            press = _event(int(p), "press", t_rel, img_ids, force_mag, widths)
            cycles.append({"cycle_idx": cycle_idx, "start_idx": int(p),
                           "end_idx": int(p) + 1, "grasp": None,
                           "release": None, "presses": [press]})
            events.append(press)

    events.sort(key=lambda e: e["row_idx"])
    summary = {
        "has_gripper": has_gripper,
        "has_force": has_force,
        "force_baseline_n": baseline,
        "force_peak_min_n": force_peak_min_n,
        "n_cycles": len(cycles),
        "n_grasps": sum(c["grasp"] is not None for c in cycles),
        "n_releases": sum(c["release"] is not None for c in cycles),
        "n_presses": sum(len(c["presses"]) for c in cycles),
        "n_force_peaks_raw": int(len(peaks)),
        "force_cluster_gap_s": force_cluster_gap_s,
    }
    return events, cycles, summary


def role_events(cycles: List[Dict]) -> List[Dict]:
    """Choose role-seeding frames while preserving each interaction cycle."""
    out = []
    for cycle in cycles:
        if cycle["grasp"] is not None:
            # Seed early in the closed run. A closed interval can include both
            # carrying and placement, so its midpoint is not a semantic grasp
            # event: t004 cycle 2 holds the nut here and has already threaded
            # it onto the pegboard by mid-hold. Twenty percent is past the
            # threshold transition while remaining in the measured carry phase.
            idx = cycle["start_idx"] + max(
                1, int(0.20 * (cycle["end_idx"] - cycle["start_idx"])))
            out.append({"role": "grasped", "cycle_idx": cycle["cycle_idx"],
                        "row_idx": idx, "event": "early_hold"})
        if cycle["presses"]:
            press = max(cycle["presses"],
                        key=lambda e: e["force_mag_n"] or float("-inf"))
            out.append({"role": "contact_receiver",
                        "cycle_idx": cycle["cycle_idx"],
                        "row_idx": press["row_idx"], "event": "press"})
    return out
