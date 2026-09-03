"""Score automatic selections against reference tracks by recording group.

This evaluator never tunes the selector. It reports precision and coverage
separately and treats manual overrides as non-automatic abstentions.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict

import cv2
import numpy as np
import yaml

from event_utils import mask_from_overlay
from select_objects import Proposal, merged_grasp_proposals

COLORS = {
    "grasped": (0, 255, 0), "contact_receiver": (255, 0, 255),
    "tool_contact": (0, 165, 255), "charger_contact": (0, 215, 255),
}
RGB_DIR = "zed_zed_node_rgb_color_rect_image_compressed"


def role_category(role):
    return "grasped" if role == "grasped" else "contact"


def iou(a, b):
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 0.0


def selected_mask(report_dir, selected):
    relative = selected.get(
        "proposal_cache_path",
        os.path.join("proposal_cache", f"{selected['img_id']}.npz"))
    cache = os.path.normpath(os.path.join(report_dir, relative))
    if not os.path.exists(cache):
        return None
    data = np.load(cache, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    target = [round(float(v)) for v in selected["seed_box_xyxy"]]
    for index, item in enumerate(meta):
        if [round(float(v)) for v in item["bbox"]] == target:
            return data["masks"][index].astype(bool)
    # Grasped selections may be local unions of cached SAM proposals. Rebuild
    # those deterministic unions for evaluation rather than treating them as
    # missing predictions.
    proposals = [Proposal(data["masks"][i].astype(bool), tuple(x["bbox"]),
                          int(x["area"]), float(x["predicted_iou"]),
                          float(x["stability"]))
                 for i, x in enumerate(meta)]
    merged = merged_grasp_proposals(proposals, data["masks"].shape[1:])
    for proposal in merged:
        if [round(float(v)) for v in proposal.bbox] == target:
            return proposal.mask
    return None


def declared_reference_role(recording, selected):
    """The reference object this cycle is scored against, from the manifest.

    Matching by role *category* instead is what produced the version 1 defect:
    on a frame holding several contact objects it returned whichever overlay
    recovered first, scoring the selector against an object the robot was not
    touching. An undeclared cycle is reported, never guessed.
    """
    for entry in recording.get("ground_truth", []):
        if (entry["selector_role"] == selected["role"] and
                int(entry["cycle_idx"]) == int(selected["cycle_idx"])):
            return entry["reference_role"]
    return None


def reference_mask(recording, selected, side_rows):
    filename = f"{selected['img_id']}.png"
    wanted_role = declared_reference_role(recording, selected)
    if wanted_role is None:
        return None, None
    candidates = [r for r in side_rows if r.get("img_filename") == filename and
                  r.get("role") == wanted_role]
    if not candidates:
        return None, None
    rgb = os.path.join(recording["trial"], RGB_DIR, filename)
    for row in candidates:
        role = row["role"]
        mask = mask_from_overlay(row["overlay_path"], rgb, COLORS[role])
        if mask is not None and mask.any():
            return mask, role
    # Coarse but explicit fallback for old overlays that cannot be recovered.
    image = cv2.imread(rgb)
    if image is None:
        return None, None
    row = candidates[0]
    try:
        x0, y0, x1, y1 = [int(float(row[k])) for k in
                          ("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1")]
    except (KeyError, ValueError):
        return None, None
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[y0:y1 + 1, x0:x1 + 1] = True
    return mask, row["role"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="config/evaluation_manifest.yaml")
    ap.add_argument("--report", action="append", required=True,
                    help="recording_id:path/to/selection_report.json")
    ap.add_argument("--out", default="figures/deliverable_selection_eval.json")
    args = ap.parse_args()
    manifest = yaml.safe_load(open(args.manifest))
    recordings = {r["id"]: r for r in manifest["recordings"]}
    rows, aggregates = [], defaultdict(lambda: {"eligible": 0, "accepted": 0,
                                                 "correct": 0, "groups": set()})

    for spec in args.report:
        rec_id, report_path = spec.split(":", 1)
        rec = recordings[rec_id]
        side = list(csv.DictReader(open(rec["reference_sidecar"])))
        report = json.load(open(report_path))
        report_dir = os.path.dirname(report_path)
        for selected in report.get("selections", []):
            declared = declared_reference_role(rec, selected)
            gt, gt_role = reference_mask(rec, selected, side)
            if gt is None:
                if declared is not None:
                    # The manifest says this cycle is scorable but its reference
                    # mask could not be recovered on the selected frame. That is
                    # a gap in the evaluation set, not a selector abstention, so
                    # it is reported instead of silently dropped.
                    rows.append({"recording": rec_id,
                                 "group": rec["independent_group"],
                                 "role": selected["role"],
                                 "reference_role": declared,
                                 "cycle_idx": selected["cycle_idx"],
                                 "automatic": bool(selected.get("automatic", True)),
                                 "accepted": None, "iou": None, "correct": False,
                                 "excluded": "ground_truth_missing"})
                continue
            category = role_category(selected["role"])
            agg = aggregates[category]
            agg["eligible"] += 1
            agg["groups"].add(rec["independent_group"])
            automatic = bool(selected.get("automatic", True))
            accepted = selected.get("status") == "accepted" and automatic
            overlap = None
            correct = False
            if accepted:
                pred = selected_mask(report_dir, selected)
                if pred is not None:
                    overlap = iou(pred, gt)
                    correct = overlap >= float(manifest["selection_iou_threshold"])
                agg["accepted"] += 1
                agg["correct"] += int(correct)
            rows.append({"recording": rec_id, "group": rec["independent_group"],
                         "role": selected["role"], "reference_role": gt_role,
                         "cycle_idx": selected["cycle_idx"], "automatic": automatic,
                         "accepted": accepted, "iou": overlap, "correct": correct})

    summary = {}
    passed = True
    for role, agg in aggregates.items():
        precision = agg["correct"] / agg["accepted"] if agg["accepted"] else 0.0
        coverage = agg["accepted"] / agg["eligible"] if agg["eligible"] else 0.0
        enough_groups = len(agg["groups"]) >= 3
        role_pass = (precision >= float(manifest["minimum_precision"]) and
                     coverage >= float(manifest["minimum_coverage"]) and enough_groups)
        passed &= role_pass
        summary[role] = {"eligible": agg["eligible"], "accepted": agg["accepted"],
                         "correct": agg["correct"], "precision": precision,
                         "coverage": coverage, "independent_groups": len(agg["groups"]),
                         "evidence_sufficient": enough_groups, "passed": role_pass}
    if not {"grasped", "contact"}.issubset(summary):
        passed = False
    output = {"protocol": manifest["protocol"], "thresholds": {
                  "iou": manifest["selection_iou_threshold"],
                  "precision": manifest["minimum_precision"],
                  "coverage": manifest["minimum_coverage"]},
              "passed": passed, "summary": summary, "rows": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(json.dumps(output, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
