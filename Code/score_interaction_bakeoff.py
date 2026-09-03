"""Map external model evidence to frozen SAM proposals and evaluate it locally."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

from evaluate_selection import reference_mask, role_category
from select_objects import Proposal, merged_grasp_proposals


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_proposals(path: Path, role: str) -> list[Proposal]:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    proposals = [Proposal(data["masks"][i].astype(bool), tuple(item["bbox"]),
                          int(item["area"]), float(item["predicted_iou"]),
                          float(item["stability"]))
                 for i, item in enumerate(meta)]
    if role == "grasped":
        proposals += merged_grasp_proposals(proposals, data["masks"].shape[1:])
    return proposals


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 0.0


def box_mask(shape: tuple[int, int], box) -> np.ndarray:
    h, w = shape
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    x0, x1 = sorted((max(0, min(w - 1, x0)), max(0, min(w - 1, x1))))
    y0, y1 = sorted((max(0, min(h - 1, y0)), max(0, min(h - 1, y1))))
    result = np.zeros(shape, dtype=bool)
    result[y0:y1 + 1, x0:x1 + 1] = True
    return result


def best_proposal(proposals: list[Proposal], evidence: np.ndarray):
    ranked = sorted(((iou(p.mask, evidence), p) for p in proposals),
                    key=lambda item: item[0], reverse=True)
    return ranked[0] if ranked else (0.0, None)


def hoi_evidence(record: dict, role: str):
    detections = record.get("detections", [])
    hf = record.get("hf", [])
    fs = record.get("fs", [])
    candidates = []
    if role == "grasped":
        for relation in hf:
            a, b = int(relation["a"]), int(relation["b"])
            if (0 <= a < len(detections) and 0 <= b < len(detections) and
                    int(detections[a]["class_id"]) == 0 and
                    int(detections[b]["class_id"]) == 1):
                candidates.append((float(relation["prob"]),
                                   float(detections[b]["score"]),
                                   detections[b]["box"], {"hf": relation}))
    else:
        linked_first = {int(relation["b"]): float(relation["prob"])
                        for relation in hf
                        if 0 <= int(relation["b"]) < len(detections)}
        for relation in fs:
            a, b = int(relation["a"]), int(relation["b"])
            if (a in linked_first and 0 <= b < len(detections) and
                    int(detections[a]["class_id"]) == 1 and
                    int(detections[b]["class_id"]) == 2):
                chain_prob = min(linked_first[a], float(relation["prob"]))
                candidates.append((chain_prob, float(detections[b]["score"]),
                                   detections[b]["box"],
                                   {"hf_probability": linked_first[a],
                                    "fs": relation}))
    return max(candidates, default=None, key=lambda item: (item[0], item[1]))


def selection_from_box(case: dict, bundle: Path, record: dict | None) -> dict:
    base = {"role": case["role"], "cycle_idx": case["cycle_idx"],
            "event": "external_model", "row_idx": None,
            "img_id": case["img_id"], "automatic": True}
    evidence = hoi_evidence(record or {}, case["role"])
    if evidence is None:
        return {**base, "status": "review_required",
                "reason": "no_complete_hoi_relation_chain"}
    relation_prob, detection_score, box, relation = evidence
    proposals = load_proposals(bundle / case["proposal_cache"], case["role"])
    if not proposals:
        return {**base, "status": "review_required", "reason": "no_proposals"}
    shape = proposals[0].mask.shape
    match, proposal = best_proposal(proposals, box_mask(shape, box))
    return {**base, "status": "accepted", "reason": "hoi_relation",
            "score": relation_prob, "detection_score": detection_score,
            "evidence_box_xyxy": box, "proposal_match_iou": match,
            "relation": relation, "seed_box_xyxy": list(proposal.bbox)}


def selection_from_mask(case: dict, bundle: Path, mask_path: Path) -> dict:
    base = {"role": case["role"], "cycle_idx": case["cycle_idx"],
            "event": "external_model", "row_idx": None,
            "img_id": case["img_id"], "automatic": True}
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return {**base, "status": "review_required", "reason": "missing_mask"}
    proposals = load_proposals(bundle / case["proposal_cache"], case["role"])
    if not proposals:
        return {**base, "status": "review_required", "reason": "no_proposals"}
    if mask.shape != proposals[0].mask.shape:
        mask = cv2.resize(mask, (proposals[0].mask.shape[1],
                                proposals[0].mask.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    match, proposal = best_proposal(proposals, mask > 127)
    if match < 0.50:
        return {**base, "status": "review_required",
                "reason": "foreground_proposal_iou_below_0.50",
                "proposal_match_iou": match}
    return {**base, "status": "accepted", "reason": "distinctnet_foreground",
            "score": match, "proposal_match_iou": match,
            "seed_box_xyxy": list(proposal.bbox)}


def write_reports(out: Path, model: str, cases: list[dict], selections: list[dict],
                  bundle: Path) -> list[tuple[str, Path]]:
    by_recording = defaultdict(list)
    for case, selection in zip(cases, selections):
        by_recording[case["recording_id"]].append((case, selection))
    reports = []
    for recording, values in sorted(by_recording.items()):
        report_dir = out / model / recording
        report_dir.mkdir(parents=True, exist_ok=True)
        packed = []
        for case, selection in values:
            item = dict(selection)
            item["proposal_cache_path"] = os.path.relpath(
                bundle / case["proposal_cache"], report_dir)
            packed.append(item)
        report = {"schema_version": "1.0", "trial": recording,
                  "status": "experimental_bakeoff", "automatic": True,
                  "model": model, "selections": packed,
                  "policy": {"low_confidence": "abstain",
                             "production_sidecar_allowed": False}}
        path = report_dir / "selection_report.json"
        path.write_text(json.dumps(report, indent=2))
        reports.append((recording, path))
    return reports


def run_evaluator(manifest: Path, reports: list[tuple[str, Path]], out: Path):
    command = [sys.executable, "Code/evaluate_selection.py",
               "--manifest", str(manifest), "--out", str(out)]
    for recording, path in reports:
        command.extend(["--report", f"{recording}:{path}"])
    completed = subprocess.run(command, check=False)
    if not out.exists():
        raise RuntimeError(f"evaluator failed with exit code {completed.returncode}")
    return json.loads(out.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--manifest", default="config/evaluation_manifest.yaml")
    parser.add_argument("--hoi-predictions")
    parser.add_argument("--distinct-predictions")
    parser.add_argument("--out", default="figures/interaction_bakeoff/scored")
    args = parser.parse_args()
    if not args.hoi_predictions and not args.distinct_predictions:
        parser.error("at least one prediction file is required")

    bundle = Path(args.bundle).resolve()
    manifest_path = Path(args.manifest).resolve()
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    benchmark_path = bundle / "benchmark.json"
    benchmark = json.loads(benchmark_path.read_text())
    cases = benchmark["cases"]
    benchmark_hash = sha256(benchmark_path)
    verdicts = {}

    if args.hoi_predictions:
        path = Path(args.hoi_predictions).resolve()
        payload = json.loads(path.read_text())
        provenance = payload.get("bakeoff_provenance", {})
        if provenance.get("benchmark_sha256") != benchmark_hash:
            raise ValueError("HOI-DETR predictions do not match this benchmark")
        images = {Path(item["file_name"]).stem: item
                  for item in payload.get("images", [])}
        selections = [selection_from_box(case, bundle, images.get(case["case_id"]))
                      for case in cases]
        reports = write_reports(out, "hoi_detr", cases, selections, bundle)
        evaluation = run_evaluator(
            manifest_path, reports, out / "hoi_detr_evaluation.json")
        verdicts["hoi_detr"] = {"target_roles": ["grasped", "contact"],
                                "passed": evaluation.get("passed", False),
                                "summary": evaluation.get("summary", {})}

    if args.distinct_predictions:
        path = Path(args.distinct_predictions).resolve()
        payload = json.loads(path.read_text())
        if payload.get("benchmark_sha256") != benchmark_hash:
            raise ValueError("DistinctNet predictions do not match this benchmark")
        pred_root = path.parent
        case_by_id = {case["case_id"]: case for case in cases}
        for variant in ("raw", "stabilized"):
            records = {p["case_id"]: p for p in payload.get("predictions", [])
                       if p.get("variant") == variant}
            selected_cases = [case for case in cases if case["role"] == "grasped"]
            selections = []
            for case in selected_cases:
                record = records.get(case["case_id"])
                mask = pred_root / record["mask"] if record else Path("missing")
                selections.append(selection_from_mask(case, bundle, mask))
            model = f"distinctnet_{variant}"
            reports = write_reports(out, model, selected_cases, selections, bundle)
            evaluation = run_evaluator(
                manifest_path, reports, out / f"{model}_evaluation.json")
            grasp = evaluation.get("summary", {}).get("grasped", {})
            verdicts[model] = {"target_roles": ["grasped"],
                               "passed": bool(grasp.get("passed", False)),
                               "summary": evaluation.get("summary", {})}

    final = {"schema_version": "1.0", "benchmark_sha256": benchmark_hash,
             "production_integration_authorized": False, "verdicts": verdicts}
    (out / "verdict.json").write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
