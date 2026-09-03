"""Feasibility study for calibration-free contact proposal ranking.

This is not a selector result. It asks two narrower questions using only cached
SAM proposals and the already-declared per-cycle ground truth:

1. Does every contact case contain a proposal capable of IoU >= the manifest
   threshold (the proposal-pool oracle)?
2. Can one linear combination of the selector's existing calibration-free
   features plus a bounded force-anchored local-flow cue rank that proposal
   first with the frozen 0.10 margin, when weights are selected on other
   recording groups only?

An optimistic all-data fit is reported separately and explicitly as leakage.
If even that fit fails, further reweighting of the current feature family has a
measured stop condition; a new identity/geometric cue is required.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deliverable_events import detect_events, find_demo_csv, load_demo_rows
from evaluate_selection import declared_reference_role, iou, reference_mask
from select_objects import (Proposal, _image_for_row, _load_proposals,
                            border_sides, mask_iou, polygon_mask,
                            proposal_features, temporal_maps)

RGB_DIR = "zed_zed_node_rgb_color_rect_image_compressed"
FEATURES = (
    "region_overlap",
    "region_proximity",
    "sam_quality",
    "border_score",
    "appearance_novelty",
    "stationarity",
    "temporal_stability",
    "area_fraction",
    "contact_flow_contrast",
)


@dataclass
class Candidate:
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    values: np.ndarray
    ground_truth_iou: float


@dataclass
class Case:
    recording: str
    group: str
    cycle_idx: int
    candidates: List[Candidate]


def candidate_score(candidate: Candidate, weights: np.ndarray) -> float:
    return float(candidate.values @ weights)


def contact_flow_contrast(mask: np.ndarray,
                          flow: Optional[np.ndarray]) -> float:
    """Signed local motion relative to a ring around the proposal.

    The force-contact instant supplies a physical temporal anchor without a
    camera extrinsic.  Normalising proposal motion by its immediate surround
    removes much of the eye-in-hand camera motion.  Positive values mean the
    proposal moves more than its surround; negative values mean less.  This is
    evaluated here before it is allowed into the production selector.
    """
    if flow is None or not mask.any():
        return 0.0
    kernel = np.ones((31, 31), dtype=np.uint8)
    expanded = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    ring = expanded & ~mask
    if not ring.any():
        return 0.0
    local = float(np.median(flow[mask]))
    surround = float(np.median(flow[ring]))
    return float((local - surround) / max(local + surround, 1e-6))


def score_case(case: Case, weights: np.ndarray, margin_min: float,
               suppression_iou: float, correct_iou: float = 0.5) -> Dict:
    order = sorted(range(len(case.candidates)),
                   key=lambda i: candidate_score(case.candidates[i], weights),
                   reverse=True)
    if not order:
        return {"accepted": False, "correct": False, "margin": None,
                "winner_iou": None}
    winner = case.candidates[order[0]]
    winner_score = candidate_score(winner, weights)
    distinct_score = 0.0
    for index in order[1:]:
        contender = case.candidates[index]
        if mask_iou(winner.mask, contender.mask) <= suppression_iou:
            distinct_score = candidate_score(contender, weights)
            break
    margin = winner_score - distinct_score
    accepted = margin >= margin_min
    return {"accepted": bool(accepted),
            "correct": bool(accepted and winner.ground_truth_iou >= correct_iou),
            "margin": float(margin),
            "winner_iou": float(winner.ground_truth_iou),
            "winner_bbox": list(winner.bbox)}


def evaluate(cases: Sequence[Case], weights: np.ndarray, margin_min: float,
             suppression_iou: float, correct_iou: float = 0.5) -> Dict:
    rows = [score_case(case, weights, margin_min, suppression_iou, correct_iou)
            for case in cases]
    accepted = sum(row["accepted"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    return {"eligible": len(rows), "accepted": accepted, "correct": correct,
            "precision": correct / accepted if accepted else 0.0,
            "coverage": accepted / len(rows) if rows else 0.0,
            "rows": rows}


def weight_grid() -> Iterable[np.ndarray]:
    """All signed sparse linear rules, normalized so margins are comparable."""
    current = np.array([0.25, 0.40, 0.15, 0.10, 0.0, 0.0, 0.10, 0.0,
                        0.0])
    yield current / np.abs(current).sum()
    for raw in itertools.product((-1.0, 0.0, 1.0), repeat=len(FEATURES)):
        weights = np.asarray(raw, dtype=float)
        norm = float(np.abs(weights).sum())
        if norm:
            yield weights / norm


def select_weights(training: Sequence[Case], margin_min: float,
                   suppression_iou: float, precision_min: float,
                   correct_iou: float = 0.5) -> Tuple[np.ndarray, Dict]:
    best_weights = np.zeros(len(FEATURES), dtype=float)
    best_result = evaluate(training, best_weights, margin_min,
                           suppression_iou, correct_iou)
    best_key = (-1, -1.0, -1.0, float("-inf"))
    for weights in weight_grid():
        result = evaluate(training, weights, margin_min,
                          suppression_iou, correct_iou)
        feasible = result["accepted"] > 0 and result["precision"] >= precision_min
        # Maximize correct automatic coverage while obeying the precision bar;
        # deterministic L1/signed lexicographic tie-break avoids hidden tuning.
        key = (int(feasible), result["coverage"] if feasible else 0.0,
               result["precision"] if feasible else 0.0,
               -float(np.count_nonzero(weights)))
        if key > best_key:
            best_key, best_weights, best_result = key, weights.copy(), result
    return best_weights, best_result


def load_case(recording: Dict, selected: Dict, report_dir: str,
              rig: Dict) -> Optional[Case]:
    rows = load_demo_rows(find_demo_csv(recording["trial"]))
    _, cycles, _ = detect_events(rows)
    cycle = next((c for c in cycles
                  if int(c["cycle_idx"]) == int(selected["cycle_idx"])), None)
    if cycle is None:
        return None
    image_dir = os.path.join(recording["trial"], RGB_DIR)
    row_idx = int(selected["row_idx"])
    img_id, image = _image_for_row(rows, row_idx, image_dir)
    if image is None or img_id != str(selected["img_id"]):
        return None
    relative = selected.get(
        "proposal_cache_path", os.path.join("proposal_cache", f"{img_id}.npz"))
    cache = os.path.normpath(os.path.join(report_dir, relative))
    proposals = _load_proposals(cache)
    if proposals is None:
        return None

    side_rows = list(csv.DictReader(open(recording["reference_sidecar"])))
    ground_truth, _ = reference_mask(recording, selected, side_rows)
    if ground_truth is None:
        return None

    h, w = image.shape[:2]
    region = polygon_mask((h, w), rig["regions"]["contact_receiver"]["polygon"])
    _, reference = _image_for_row(rows, max(0, row_idx - 20), image_dir)
    _, partner = _image_for_row(rows, min(len(rows) - 1, row_idx + 20), image_dir)
    change, flow, bg_flow = temporal_maps(image, reference, partner)
    filt = rig["proposal_filter"]
    candidates = []
    for proposal in proposals:
        area_fraction = proposal.area / max(1, h * w)
        if not (float(filt["min_area_fraction"]) <= area_fraction <=
                float(filt["max_area_fraction"])):
            continue
        if border_sides(proposal.mask) > int(filt["max_border_sides"]):
            continue
        values = proposal_features(proposal, region, "contact_receiver",
                                   change, flow, bg_flow)
        values["area_fraction"] = float(area_fraction)
        values["contact_flow_contrast"] = contact_flow_contrast(
            proposal.mask, flow)
        candidates.append(Candidate(
            proposal.mask, proposal.bbox,
            np.array([float(values[name]) for name in FEATURES]),
            iou(proposal.mask, ground_truth)))
    return Case(recording["id"], recording["independent_group"],
                int(selected["cycle_idx"]), candidates)


def weights_doc(weights: np.ndarray) -> Dict[str, float]:
    return {name: float(value) for name, value in zip(FEATURES, weights)
            if abs(float(value)) > 1e-12}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="config/evaluation_manifest.yaml")
    ap.add_argument("--rig-profile", default="config/deliverable_rig.yaml")
    ap.add_argument("--report", action="append", required=True,
                    help="recording_id:path/to/selection_report.json")
    ap.add_argument("--out", default="figures/contact_flow_contrast_study.json")
    args = ap.parse_args()

    manifest = yaml.safe_load(open(args.manifest))
    rig = yaml.safe_load(open(args.rig_profile))
    recordings = {r["id"]: r for r in manifest["recordings"]}
    cases = []
    missing = []
    for spec in args.report:
        recording_id, report_path = spec.split(":", 1)
        report = json.load(open(report_path))
        for selected in report.get("selections", []):
            if selected.get("role") != "contact_receiver":
                continue
            if declared_reference_role(recordings[recording_id], selected) is None:
                # The selector may emit an operational contact role for a
                # recording whose manifest deliberately declares no scorable
                # contact ground truth. That is out of scope, not missing.
                continue
            case = load_case(recordings[recording_id], selected,
                             os.path.dirname(report_path), rig)
            if case is None:
                missing.append({"recording": recording_id,
                                "cycle_idx": selected.get("cycle_idx")})
            else:
                cases.append(case)

    margin_min = float(rig["regions"]["contact_receiver"]["confidence_margin_min"])
    suppression_iou = float(rig.get("duplicate_suppression_iou", 0.5))
    precision_min = float(manifest["minimum_precision"])
    coverage_min = float(manifest["minimum_coverage"])
    correct_iou = float(manifest["selection_iou_threshold"])

    oracle_rows = []
    for case in cases:
        best = max((c.ground_truth_iou for c in case.candidates), default=0.0)
        oracle_rows.append({"recording": case.recording, "group": case.group,
                            "cycle_idx": case.cycle_idx,
                            "best_pool_iou": float(best),
                            "reachable": best >= correct_iou})

    folds = []
    heldout_rows = []
    for group in sorted({case.group for case in cases}):
        training = [case for case in cases if case.group != group]
        heldout = [case for case in cases if case.group == group]
        weights, training_result = select_weights(
            training, margin_min, suppression_iou, precision_min, correct_iou)
        heldout_result = evaluate(heldout, weights, margin_min,
                                  suppression_iou, correct_iou)
        for case, row in zip(heldout, heldout_result.pop("rows")):
            heldout_rows.append({"recording": case.recording,
                                 "group": case.group,
                                 "cycle_idx": case.cycle_idx, **row})
        training_result.pop("rows", None)
        folds.append({"heldout_group": group, "weights": weights_doc(weights),
                      "training": training_result, "heldout": heldout_result})

    accepted = sum(row["accepted"] for row in heldout_rows)
    correct = sum(row["correct"] for row in heldout_rows)
    group_result = {"eligible": len(heldout_rows), "accepted": accepted,
                    "correct": correct,
                    "precision": correct / accepted if accepted else 0.0,
                    "coverage": accepted / len(heldout_rows) if heldout_rows else 0.0}
    group_result["passed"] = (
        group_result["precision"] >= precision_min and
        group_result["coverage"] >= coverage_min and
        len({case.group for case in cases}) >= 3)

    optimistic_weights, optimistic = select_weights(
        cases, margin_min, suppression_iou, precision_min, correct_iou)
    optimistic.pop("rows", None)
    optimistic["weights"] = weights_doc(optimistic_weights)
    optimistic["passed"] = (optimistic["precision"] >= precision_min and
                             optimistic["coverage"] >= coverage_min)

    output = {
        "study_type": "feature_family_feasibility_not_performance",
        "search_space": {
            "kind": "normalized_signed_linear_feature_subset_grid",
            "coefficient_values_before_l1_normalization": [-1, 0, 1],
            "rules_tested_per_fit": 3 ** len(FEATURES),
            "scope_note": "bounded test of existing features plus one physically motivated force-anchored flow cue; not a proof over every nonlinear cue"
        },
        "features": list(FEATURES),
        "thresholds": {"iou": correct_iou,
                       "margin": margin_min, "precision": precision_min,
                       "coverage": coverage_min,
                       "duplicate_suppression_iou": suppression_iou},
        "proposal_pool_oracle": {"reachable": sum(r["reachable"] for r in oracle_rows),
                                 "eligible": len(oracle_rows), "rows": oracle_rows},
        "group_separated": {**group_result, "rows": heldout_rows,
                            "folds": folds},
        "optimistic_all_data_leakage": optimistic,
        "missing_cases": missing,
    }
    if optimistic["passed"]:
        conclusion = "current_features_have_in_sample_capacity_test_group_generalization"
    else:
        conclusion = "reject_contact_flow_contrast_seek_different_identity_cue"
    output["conclusion"] = conclusion

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(output, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    if os.path.exists(args.out):
        import shutil
        shutil.copy2(args.out, args.out + ".bak")
    os.replace(tmp, args.out)
    print(json.dumps(output, indent=2))
    return 0 if group_result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
