"""Feasibility gate for calibration-free grasped-object identity.

The held object should change from scene-relative before closure to stable in
the eye-in-hand camera after closure.  This study anchors every candidate at
the first configured hold probe (20%), matches it against the later cached SAM
proposal pools, and asks whether that attachment evidence can clear the frozen
4/5, precision-1.0 development bar.  It does not modify the production
selector.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deliverable_events import detect_events, find_demo_csv, load_demo_rows
from evaluate_selection import iou, reference_mask
from select_objects import (Proposal, _image_for_row, _load_proposals,
                            border_sides, mask_iou, merged_grasp_proposals)

RGB_DIR = "zed_zed_node_rgb_color_rect_image_compressed"
FEATURES = (
    "post_mask_iou",
    "post_support",
    "centroid_stability",
    "area_stability",
    "preclose_change",
    "sam_quality",
    "border_score",
)


@dataclass
class Candidate:
    proposal: Proposal
    values: np.ndarray
    ground_truth_iou: float


@dataclass
class Case:
    recording: str
    group: str
    cycle_idx: int
    anchor_img_id: str
    candidates: List[Candidate]


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def center(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return np.array([float(np.mean(xs)), float(np.mean(ys))])


def attachment_features(anchor: Proposal,
                        later_pools: Sequence[Sequence[Proposal]],
                        preclose_change: np.ndarray) -> Dict[str, float]:
    """Physical evidence for a proposal becoming camera-attached."""
    overlaps, drifts, area_changes = [], [], []
    diag = float(np.hypot(*anchor.mask.shape))
    anchor_center = center(anchor.mask)
    for pool in later_pools:
        if not pool:
            overlaps.append(0.0)
            continue
        match = max(pool, key=lambda proposal: mask_iou(anchor.mask,
                                                        proposal.mask))
        overlap = mask_iou(anchor.mask, match.mask)
        overlaps.append(overlap)
        drifts.append(float(np.linalg.norm(center(match.mask) - anchor_center) /
                            max(diag, 1.0)))
        area_changes.append(abs(float(np.log(max(match.area, 1) /
                                             max(anchor.area, 1)))))
    support = (sum(value >= 0.25 for value in overlaps) / len(overlaps)
               if overlaps else 0.0)
    median_overlap = float(np.median(overlaps)) if overlaps else 0.0
    median_drift = float(np.median(drifts)) if drifts else 1.0
    median_area_change = (float(np.median(area_changes))
                          if area_changes else float("inf"))
    novelty = (float(np.median(preclose_change[anchor.mask]))
               if anchor.mask.any() else 0.0)
    quality = float(np.clip(0.5 * anchor.predicted_iou +
                            0.5 * anchor.stability, 0.0, 1.0))
    return {
        "post_mask_iou": median_overlap,
        "post_support": float(support),
        "centroid_stability": float(np.exp(-median_drift / 0.05)),
        "area_stability": (float(np.exp(-median_area_change))
                           if np.isfinite(median_area_change) else 0.0),
        "preclose_change": novelty,
        "sam_quality": quality,
        "border_score": 1.0 - border_sides(anchor.mask) / 4.0,
    }


def filtered_proposals(path: str, rig: Dict,
                       include_unions: bool = True) -> List[Proposal]:
    proposals = _load_proposals(path)
    if proposals is None:
        return []
    if include_unions and proposals:
        proposals = proposals + merged_grasp_proposals(
            proposals, proposals[0].mask.shape)
    filt = rig["proposal_filter"]
    image_area = proposals[0].mask.size if proposals else 1
    return [proposal for proposal in proposals
            if float(filt["min_area_fraction"]) <=
            proposal.area / image_area <= float(filt["max_area_fraction"])
            and border_sides(proposal.mask) <= int(filt["max_border_sides"])]


def weight_grid() -> Iterable[np.ndarray]:
    for raw in itertools.product((-1.0, 0.0, 1.0), repeat=len(FEATURES)):
        weights = np.asarray(raw, dtype=float)
        norm = float(np.abs(weights).sum())
        if norm:
            yield weights / norm


def score_case(case: Case, weights: np.ndarray, margin_min: float,
               suppression_iou: float, correct_iou: float) -> Dict:
    order = sorted(range(len(case.candidates)),
                   key=lambda index: float(case.candidates[index].values @ weights),
                   reverse=True)
    if not order:
        return {"accepted": False, "correct": False, "margin": None,
                "winner_iou": None}
    winner = case.candidates[order[0]]
    score = float(winner.values @ weights)
    distinct = 0.0
    for index in order[1:]:
        contender = case.candidates[index]
        if mask_iou(winner.proposal.mask,
                    contender.proposal.mask) <= suppression_iou:
            distinct = float(contender.values @ weights)
            break
    margin = score - distinct
    accepted = margin >= margin_min
    return {"accepted": bool(accepted),
            "correct": bool(accepted and
                            winner.ground_truth_iou >= correct_iou),
            "margin": float(margin),
            "winner_iou": float(winner.ground_truth_iou),
            "winner_bbox": list(winner.proposal.bbox)}


def evaluate(cases: Sequence[Case], weights: np.ndarray, margin_min: float,
             suppression_iou: float, correct_iou: float) -> Dict:
    rows = [score_case(case, weights, margin_min, suppression_iou, correct_iou)
            for case in cases]
    accepted = sum(row["accepted"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    return {"eligible": len(rows), "accepted": accepted, "correct": correct,
            "precision": correct / accepted if accepted else 0.0,
            "coverage": accepted / len(rows) if rows else 0.0,
            "rows": rows}


def select_weights(cases: Sequence[Case], margin_min: float,
                   suppression_iou: float, correct_iou: float) -> Tuple[np.ndarray, Dict]:
    best_weights = np.zeros(len(FEATURES))
    best = evaluate(cases, best_weights, margin_min, suppression_iou,
                    correct_iou)
    best_key = (-1, -1, -1.0, -1.0)
    for weights in weight_grid():
        result = evaluate(cases, weights, margin_min, suppression_iou,
                          correct_iou)
        precise = result["accepted"] > 0 and result["precision"] == 1.0
        key = (int(precise), result["correct"] if precise else 0,
               result["coverage"] if precise else 0.0,
               -float(np.count_nonzero(weights)))
        if key > best_key:
            best_key, best_weights, best = key, weights.copy(), result
    return best_weights, best


def load_case(recording: Dict, selected: Dict, report_dir: str,
              rig: Dict) -> Optional[Case]:
    probe_ids = list(selected.get("hold_probe_frames", []))
    if len(probe_ids) < 3:
        return None
    rows = load_demo_rows(find_demo_csv(recording["trial"]))
    _, cycles, _ = detect_events(rows)
    cycle = next((item for item in cycles
                  if int(item["cycle_idx"]) == int(selected["cycle_idx"])), None)
    if cycle is None:
        return None
    image_dir = os.path.join(recording["trial"], RGB_DIR)
    anchor_id = str(probe_ids[0])
    anchor_path = os.path.join(image_dir, anchor_id + ".png")
    anchor_image = cv2.imread(anchor_path)
    _, preclose = _image_for_row(rows, max(0, int(cycle["start_idx"]) - 1),
                                 image_dir)
    if anchor_image is None or preclose is None or preclose.shape != anchor_image.shape:
        return None
    gray = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2GRAY)
    pre_gray = cv2.cvtColor(preclose, cv2.COLOR_BGR2GRAY)
    change = cv2.absdiff(gray, pre_gray).astype(np.float32) / 255.0
    cache_dir = os.path.join(report_dir, "proposal_cache")
    anchor_pool = filtered_proposals(os.path.join(cache_dir, anchor_id + ".npz"),
                                     rig)
    later_pools = [filtered_proposals(os.path.join(cache_dir, str(img_id) + ".npz"),
                                      rig)
                   for img_id in probe_ids[1:]]
    if not anchor_pool or any(not pool for pool in later_pools):
        return None
    side_rows = list(csv.DictReader(open(recording["reference_sidecar"])))
    anchor_selection = dict(selected, img_id=anchor_id)
    ground_truth, _ = reference_mask(recording, anchor_selection, side_rows)
    if ground_truth is None:
        return None
    candidates = []
    for proposal in anchor_pool:
        values = attachment_features(proposal, later_pools, change)
        candidates.append(Candidate(
            proposal=proposal,
            values=np.asarray([values[name] for name in FEATURES], dtype=float),
            ground_truth_iou=iou(proposal.mask, ground_truth)))
    return Case(recording["id"], recording["independent_group"],
                int(selected["cycle_idx"]), anchor_id, candidates)


def weights_doc(weights: np.ndarray) -> Dict[str, float]:
    return {name: float(value) for name, value in zip(FEATURES, weights)
            if abs(float(value)) > 1e-12}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="config/evaluation_manifest.yaml")
    parser.add_argument("--rig-profile", default="config/deliverable_rig.yaml")
    parser.add_argument("--report", action="append", required=True,
                        help="recording_id:path/to/selection_report.json")
    parser.add_argument("--out", default="figures/grasp_attachment_study.json")
    args = parser.parse_args()

    manifest = yaml.safe_load(open(args.manifest))
    rig = yaml.safe_load(open(args.rig_profile))
    recordings = {item["id"]: item for item in manifest["recordings"]}
    cases, missing = [], []
    for spec in args.report:
        recording_id, report_path = spec.split(":", 1)
        report = json.load(open(report_path))
        for selected in report.get("selections", []):
            if selected.get("role") != "grasped":
                continue
            declared = any(entry["selector_role"] == "grasped" and
                           int(entry["cycle_idx"]) == int(selected["cycle_idx"])
                           for entry in recordings[recording_id].get("ground_truth", []))
            if not declared:
                continue
            case = load_case(recordings[recording_id], selected,
                             os.path.dirname(report_path), rig)
            if case is None:
                missing.append({"recording": recording_id,
                                "cycle_idx": selected.get("cycle_idx")})
            else:
                cases.append(case)

    margin_min = float(rig["regions"]["grasped"]["confidence_margin_min"])
    suppression_iou = float(rig.get("duplicate_suppression_iou", 0.5))
    correct_iou = float(manifest["selection_iou_threshold"])
    oracle_rows = [{"recording": case.recording, "group": case.group,
                    "cycle_idx": case.cycle_idx,
                    "best_pool_iou": max((candidate.ground_truth_iou
                                          for candidate in case.candidates),
                                         default=0.0)}
                   for case in cases]

    optimistic_weights, optimistic = select_weights(
        cases, margin_min, suppression_iou, correct_iou)
    optimistic_rows = optimistic.pop("rows")
    optimistic["weights"] = weights_doc(optimistic_weights)
    optimistic["passed_feasibility"] = (
        optimistic["correct"] >= 4 and optimistic["precision"] == 1.0)

    folds, heldout_rows = [], []
    for group in sorted({case.group for case in cases}):
        training = [case for case in cases if case.group != group]
        heldout = [case for case in cases if case.group == group]
        weights, training_result = select_weights(
            training, margin_min, suppression_iou, correct_iou)
        heldout_result = evaluate(heldout, weights, margin_min,
                                  suppression_iou, correct_iou)
        for case, row in zip(heldout, heldout_result.pop("rows")):
            heldout_rows.append({"recording": case.recording,
                                 "group": case.group,
                                 "cycle_idx": case.cycle_idx, **row})
        training_result.pop("rows", None)
        folds.append({"heldout_group": group,
                      "weights": weights_doc(weights),
                      "training": training_result,
                      "heldout": heldout_result})
    accepted = sum(row["accepted"] for row in heldout_rows)
    correct = sum(row["correct"] for row in heldout_rows)
    grouped = {"eligible": len(heldout_rows), "accepted": accepted,
               "correct": correct,
               "precision": correct / accepted if accepted else 0.0,
               "coverage": accepted / len(heldout_rows) if heldout_rows else 0.0,
               "rows": heldout_rows, "folds": folds}
    grouped["passed"] = (correct >= 4 and accepted == correct and
                         len({case.group for case in cases}) >= 3)

    output = {
        "study_type": "grasp_attachment_feasibility_not_production",
        "provenance": {
            "manifest_sha256": file_sha256(args.manifest),
            "rig_profile_sha256": file_sha256(args.rig_profile),
            "anchor": "first_hold_probe_20_percent",
            "later_probes": ["27.5_percent", "35_percent"],
        },
        "features": list(FEATURES),
        "search_space": {"kind": "normalized_signed_linear_subset_grid",
                         "rules_tested_per_fit": 3 ** len(FEATURES) - 1},
        "thresholds": {"iou": correct_iou, "margin": margin_min,
                       "required_correct": 4, "required_precision": 1.0,
                       "duplicate_suppression_iou": suppression_iou},
        "proposal_pool_oracle": {"reachable": sum(
            row["best_pool_iou"] >= correct_iou for row in oracle_rows),
            "eligible": len(oracle_rows), "rows": oracle_rows},
        "optimistic_all_data_feasibility": {**optimistic,
                                             "rows": optimistic_rows},
        "group_separated": grouped,
        "missing_cases": missing,
    }
    if not optimistic["passed_feasibility"]:
        output["conclusion"] = "stop_attachment_features_lack_capacity"
    elif not grouped["passed"]:
        output["conclusion"] = "attachment_has_capacity_but_fails_group_separation"
    else:
        output["conclusion"] = "attachment_clears_local_integration_gate"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(output, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    if os.path.exists(args.out):
        shutil.copy2(args.out, args.out + ".bak")
    os.replace(tmp, args.out)
    print(json.dumps({"proposal_pool_oracle": output["proposal_pool_oracle"],
                      "optimistic": output["optimistic_all_data_feasibility"],
                      "group_separated": grouped,
                      "missing_cases": missing,
                      "conclusion": output["conclusion"]}, indent=2))
    return 0 if grouped["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
