"""Calibration-free automatic object selection for Deliverable A.

The selector ranks SAM proposals using fixed-rig image regions and temporal
evidence. It returns box prompts, records every score, and safely abstains when
the winner is weak or insufficiently separated from the runner-up.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deliverable_events import (IMG, detect_events, find_demo_csv,
                                load_demo_rows, role_events)

RGB_DIR = "zed_zed_node_rgb_color_rect_image_compressed"
DEPTH_COL = "zed.zed_node.depth.depth_registered.compressedDepth"
DEPTH_DIR = "zed_zed_node_depth_depth_registered_compressedDepth"

ROLE_COLORS = {
    "grasped": (0, 255, 0),
    "contact_receiver": (255, 0, 255),
}


def backup_if_exists(path: str) -> None:
    if os.path.exists(path):
        shutil.copy2(path, path + ".bak")


def write_json_safely(path: str, payload: Dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2,
                  default=lambda value: value.item()
                  if isinstance(value, np.generic) else str(value))
        f.flush()
        os.fsync(f.fileno())
    backup_if_exists(path)
    os.replace(tmp, path)


@dataclass
class Proposal:
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    area: int
    predicted_iou: float
    stability: float


def polygon_mask(shape: Tuple[int, int], polygon: Iterable[Iterable[float]]) -> np.ndarray:
    """Rasterize a normalized polygon for an image ``(H, W)``."""
    h, w = shape
    pts = np.array([[round(float(x) * (w - 1)), round(float(y) * (h - 1))]
                    for x, y in polygon], dtype=np.int32)
    out = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(out, [pts], 1)
    return out.astype(bool)


def mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def interior_point(mask: np.ndarray) -> Tuple[float, float]:
    """Point maximally inside the mask; unlike a centroid, it cannot hit a hole."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return float(x), float(y)


def border_sides(mask: np.ndarray) -> int:
    return int(mask[0].any()) + int(mask[-1].any()) + \
        int(mask[:, 0].any()) + int(mask[:, -1].any())


def proximity_score(mask: np.ndarray, region: np.ndarray) -> float:
    """One inside the region; otherwise decays with pixel distance to it."""
    if (mask & region).any():
        return 1.0
    inv = (~mask).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    d = float(np.min(dist[region])) if region.any() else float("inf")
    diag = float(np.hypot(*mask.shape))
    return float(np.exp(-d / max(1.0, 0.08 * diag)))


def temporal_maps(current: np.ndarray, reference: Optional[np.ndarray],
                  partner: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    change = flow_mag = None
    bg_flow = 0.0
    gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    if reference is not None and reference.shape == current.shape:
        ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        change = cv2.absdiff(gray, ref).astype(np.float32) / 255.0
    if partner is not None and partner.shape == current.shape:
        nxt = cv2.cvtColor(partner, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(gray, nxt, None, 0.5, 3, 21,
                                            3, 5, 1.2, 0)
        flow_mag = np.linalg.norm(flow, axis=2)
        bg_flow = float(np.median(flow_mag))
    return change, flow_mag, bg_flow


def proposal_features(proposal: Proposal, region: np.ndarray, role: str,
                      change: Optional[np.ndarray] = None,
                      flow: Optional[np.ndarray] = None,
                      bg_flow: float = 0.0) -> Dict[str, float]:
    m = proposal.mask
    inter = int((m & region).sum())
    # Containment alone made every tiny fragment wholly inside the broad rig
    # polygon score 1.0. IoU rewards proposals that cover a useful fraction
    # of the role region while penalising both fragments and frame-sized
    # background masks.
    union = int((m | region).sum())
    overlap = inter / max(1, union)
    proximity = proximity_score(m, region)
    sam_quality = float(np.clip(0.5 * proposal.predicted_iou +
                                0.5 * proposal.stability, 0.0, 1.0))
    border = 1.0 - border_sides(m) / 4.0
    novelty = float(np.median(change[m])) if change is not None and m.any() else 0.0
    if flow is not None and m.any() and bg_flow > 0.25:
        local = float(np.median(flow[m]))
        stationarity = float(np.exp(-local / max(bg_flow, 1e-6)))
    else:
        stationarity = 0.5
    # A stable contact receiver should not be a transient difference region.
    temporal_stability = 1.0 - novelty

    if role == "grasped":
        score = (0.35 * overlap + 0.15 * proximity + 0.20 * stationarity +
                 0.10 * novelty + 0.15 * sam_quality + 0.05 * border)
    else:
        score = (0.25 * overlap + 0.40 * proximity +
                 0.10 * temporal_stability + 0.15 * sam_quality +
                 0.10 * border)
    return {
        "score": float(score), "region_overlap": float(overlap),
        "region_proximity": float(proximity), "sam_quality": sam_quality,
        "border_score": float(border), "appearance_novelty": novelty,
        "stationarity": stationarity,
        "temporal_stability": temporal_stability,
    }


def rank_proposals(proposals: List[Proposal], region: np.ndarray, role: str,
                   config: Dict, change=None, flow=None,
                   bg_flow: float = 0.0) -> List[Tuple[Proposal, Dict[str, float]]]:
    filt = config["proposal_filter"]
    image_area = int(region.size)
    ranked = []
    for proposal in proposals:
        af = proposal.area / max(1, image_area)
        if not (float(filt["min_area_fraction"]) <= af <=
                float(filt["max_area_fraction"])):
            continue
        if border_sides(proposal.mask) > int(filt["max_border_sides"]):
            continue
        features = proposal_features(proposal, region, role, change, flow, bg_flow)
        features["area_fraction"] = float(af)
        ranked.append((proposal, features))
    return sorted(ranked, key=lambda item: item[1]["score"], reverse=True)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 0.0


def distinct_runner_up(ranked: List[Tuple[Proposal, Dict[str, float]]],
                       suppression_iou: float) -> Tuple[float, int]:
    """Best score among candidates that are a different object from the winner.

    SAM returns the same object at several granularities, so the raw runner-up
    is routinely a nested view of the winner: on lfdws_t001_depth cycle 2 the
    top three candidates overlap the winner at IoU 0.66 and 0.71 and the margin
    collapses to 0.001. Measuring separation against those makes the margin a
    report on mask granularity rather than on object ambiguity, and abstains on
    exactly the cases where the selector is most certain. Overlapping duplicates
    are suppressed first; the margin is then the gap to the best genuinely
    different candidate.
    """
    winner = ranked[0][0]
    for proposal, features in ranked[1:]:
        if mask_iou(winner.mask, proposal.mask) <= suppression_iou:
            return float(features["score"]), int(len(ranked))
    return 0.0, int(len(ranked))


def choose(ranked: List[Tuple[Proposal, Dict[str, float]]], role_cfg: Dict,
           suppression_iou: float = 0.5) -> Dict:
    if not ranked:
        return {"status": "review_required", "reason": "no_valid_proposals"}
    best, features = ranked[0]
    raw_second = ranked[1][1]["score"] if len(ranked) > 1 else 0.0
    second, _ = distinct_runner_up(ranked, suppression_iou)
    margin = float(features["score"] - second)
    accepted = (features["score"] >= float(role_cfg["min_score"]) and
                margin >= float(role_cfg["confidence_margin_min"]))
    x, y = interior_point(best.mask)
    return {
        "status": "accepted" if accepted else "review_required",
        "reason": None if accepted else "low_score_or_margin",
        "score": features["score"], "runner_up_score": float(raw_second),
        "distinct_runner_up_score": float(second),
        "duplicate_suppression_iou": float(suppression_iou),
        "confidence_margin": margin, "features": features,
        "seed_point_xy": [x, y], "seed_box_xyxy": list(best.bbox),
        "mask_px": best.area,
    }


def _load_proposals(cache_path: str, expected_provenance: Optional[Dict] = None
                    ) -> Optional[List[Proposal]]:
    if not os.path.exists(cache_path):
        return None
    data = np.load(cache_path, allow_pickle=False)
    if expected_provenance is not None:
        if "provenance" not in data.files:
            return None
        actual = json.loads(str(data["provenance"]))
        if actual != expected_provenance:
            return None
    masks, meta = data["masks"].astype(bool), json.loads(str(data["meta"]))
    return [Proposal(mask=masks[i], bbox=tuple(m["bbox"]), area=int(m["area"]),
                     predicted_iou=float(m["predicted_iou"]),
                     stability=float(m["stability"]))
            for i, m in enumerate(meta)]


def _save_proposals(cache_path: str, proposals: List[Proposal],
                    provenance: Optional[Dict] = None) -> None:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    masks = (np.stack([p.mask for p in proposals]).astype(np.uint8)
             if proposals else np.empty((0, 0, 0), dtype=np.uint8))
    meta = [{"bbox": list(p.bbox), "area": p.area,
             "predicted_iou": p.predicted_iou, "stability": p.stability}
            for p in proposals]
    np.savez_compressed(cache_path, masks=masks, meta=json.dumps(meta),
                        provenance=json.dumps(provenance or {}, sort_keys=True))


def proposal_provenance(args) -> Dict:
    stat = os.stat(args.ckpt)
    return {
        "cache_version": 1,
        "model": args.model,
        "points_per_side": args.points_per_side,
        "pred_iou_thresh": 0.85,
        "stability_score_thresh": 0.90,
        "min_mask_region_area": 400,
        "checkpoint": os.path.basename(args.ckpt),
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
    }


def _generate(image: np.ndarray, generator) -> List[Proposal]:
    anns = generator.generate(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    out = []
    for ann in anns:
        mask = np.asarray(ann["segmentation"], dtype=bool)
        if not mask.any():
            continue
        out.append(Proposal(mask=mask, bbox=mask_bbox(mask), area=int(mask.sum()),
                            predicted_iou=float(ann.get("predicted_iou", 0.5)),
                            stability=float(ann.get("stability_score", 0.5))))
    return out


def _parse_overrides(values: List[str]) -> Dict[Tuple[str, int], List[float]]:
    out = {}
    for value in values:
        role, cycle, coords = value.split(":", 2)
        box = [float(v) for v in coords.split(",")]
        if len(box) != 4:
            raise ValueError(f"override needs x0,y0,x1,y1: {value}")
        out[(role, int(cycle))] = box
    return out


def _image_for_row(rows, row_idx: int, img_dir: str) -> Tuple[str, Optional[np.ndarray]]:
    img_id = str(rows[int(np.clip(row_idx, 0, len(rows) - 1))][IMG])
    return img_id, cv2.imread(os.path.join(img_dir, f"{img_id}.png"))


def hold_probe_indices(cycle: Dict, role: str, target_idx: int) -> List[int]:
    """Frames used to establish a stable grasp seed across the closed hold."""
    if role != "grasped":
        return [int(target_idx)]
    start, end = int(cycle["start_idx"]), int(cycle["end_idx"])
    span = max(1, end - start)
    return sorted({max(start, min(end - 1, start + int(span * f)))
                   for f in (0.20, 0.275, 0.35)})


def center_xy(box: List[float]) -> np.ndarray:
    return np.array([(float(box[0]) + float(box[2])) * 0.5,
                     (float(box[1]) + float(box[3])) * 0.5])


def merged_grasp_proposals(proposals: List[Proposal], shape: Tuple[int, int],
                           radius_fraction: float = 0.05) -> List[Proposal]:
    """Add locally coherent unions for fragmented held objects.

    SAM can split a textured or partially occluded carried object into several
    adjacent masks. Unions are formed only within a small image-space radius;
    the originals remain available, and the normal area/border filters still
    reject frame-sized unions.
    """
    if len(proposals) < 2:
        return []
    h, w = shape
    radius = float(radius_fraction * np.hypot(w, h))
    centers = [center_xy(list(p.bbox)) for p in proposals]
    out, seen = [], set()
    for i, proposal in enumerate(proposals):
        indices = [j for j, c in enumerate(centers)
                   if np.linalg.norm(c - centers[i]) <= radius]
        if len(indices) < 2:
            continue
        mask = np.any(np.stack([proposals[j].mask for j in indices]), axis=0)
        box = mask_bbox(mask)
        key = (tuple(box), int(mask.sum()))
        if key in seen:
            continue
        seen.add(key)
        quality = float(np.mean([proposals[j].predicted_iou for j in indices]))
        stability = float(np.mean([proposals[j].stability for j in indices]))
        out.append(Proposal(mask=mask, bbox=box, area=int(mask.sum()),
                            predicted_iou=quality, stability=stability))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rig-profile", default="config/deliverable_rig.yaml")
    ap.add_argument("--ckpt", default="sam_vit_h_4b8939.pth")
    ap.add_argument("--model", default="vit_h")
    ap.add_argument("--points-per-side", type=int, default=24)
    ap.add_argument("--proposal-cache", default=None)
    ap.add_argument("--override", action="append", default=[],
                    help="role:cycle:x0,y0,x1,y1; marks that selection manual")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cache_dir = args.proposal_cache or os.path.join(args.out, "proposal_cache")
    profile = yaml.safe_load(open(args.rig_profile))
    rows = load_demo_rows(find_demo_csv(args.trial))
    events, cycles, event_summary = detect_events(rows)
    targets = role_events(cycles)
    img_dir = os.path.join(args.trial, RGB_DIR)
    overrides = _parse_overrides(args.override)
    proposal_config = proposal_provenance(args)

    generator = None
    results, panels = [], []
    for target in targets:
        role, cycle_idx, idx = target["role"], target["cycle_idx"], target["row_idx"]
        cycle = next(c for c in cycles if c["cycle_idx"] == cycle_idx)
        probe_indices = hold_probe_indices(cycle, role, idx)
        img_id, image = _image_for_row(rows, idx, img_dir)
        record = {**target, "img_id": img_id, "automatic": True}
        if image is None:
            record.update(status="review_required", reason="missing_event_image")
            results.append(record)
            continue
        override = overrides.get((role, cycle_idx))
        if override is not None:
            x0, y0, x1, y1 = override
            record.update(status="accepted", reason="manual_override",
                          automatic=False, score=None, confidence_margin=None,
                          seed_box_xyxy=[x0, y0, x1, y1],
                          seed_point_xy=[0.5 * (x0 + x1), 0.5 * (y0 + y1)])
            results.append(record)
            continue

        probes = []
        for probe_idx in probe_indices:
            probe_img_id, probe_image = _image_for_row(rows, probe_idx, img_dir)
            if probe_image is None:
                continue
            h, w = probe_image.shape[:2]
            cache = os.path.join(cache_dir, f"{probe_img_id}.npz")
            proposals = _load_proposals(cache, proposal_config)
            if proposals is None:
                if generator is None:
                    import torch
                    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    print(f"[load] SAM {args.model} on {device}", flush=True)
                    model = sam_model_registry[args.model](checkpoint=args.ckpt).to(device)
                    generator = SamAutomaticMaskGenerator(
                        model, points_per_side=args.points_per_side,
                        pred_iou_thresh=0.85, stability_score_thresh=0.90,
                        min_mask_region_area=400)
                print(f"[proposals] {role}/cycle{cycle_idx} {probe_img_id}", flush=True)
                proposals = _generate(probe_image, generator)
                _save_proposals(cache, proposals, proposal_config)
            rank_proposals_input = proposals
            if role == "grasped":
                rank_proposals_input = proposals + merged_grasp_proposals(
                    proposals, (h, w))

            ref_idx = max(0, cycle["start_idx"] - 1) if role == "grasped" else max(0, probe_idx - 20)
            partner_idx = (min(len(rows) - 1,
                               cycle["start_idx"] + int(0.65 * max(1, cycle["end_idx"] - cycle["start_idx"])))
                           if role == "grasped" else min(len(rows) - 1, probe_idx + 20))
            _, reference = _image_for_row(rows, ref_idx, img_dir)
            _, partner = _image_for_row(rows, partner_idx, img_dir)
            change, flow, bg_flow = temporal_maps(probe_image, reference, partner)
            region = polygon_mask((h, w), profile["regions"][role]["polygon"])
            ranked = rank_proposals(rank_proposals_input, region, role, profile,
                                    change, flow, bg_flow)
            picked = choose(ranked, profile["regions"][role],
                            float(profile.get("duplicate_suppression_iou", 0.5)))
            probes.append({"row_idx": probe_idx, "img_id": probe_img_id,
                           "image": probe_image, "ranked": ranked,
                           "picked": picked,
                           "cache_path": os.path.relpath(cache, args.out)})

        if not probes:
            record.update(status="review_required", reason="no_valid_probe_frames")
            results.append(record)
            continue

        # A grasp seed is accepted only when its per-frame confidence is met
        # and the same spatial winner is supported across multiple hold
        # frames. Hold consensus is a consistency gate, not an identity claim.
        chosen_probe = max(probes, key=lambda p: p["picked"].get("score", 0.0))
        picked = dict(chosen_probe["picked"])
        if role == "grasped" and len(probes) >= 2 and "seed_box_xyxy" in picked:
            _, w = chosen_probe["image"].shape[:2]
            h = chosen_probe["image"].shape[0]
            chosen_center = center_xy(picked["seed_box_xyxy"])
            radius = 0.15 * float(np.hypot(w, h))
            support = sum(
                "seed_box_xyxy" in p["picked"] and
                np.linalg.norm(center_xy(p["picked"]["seed_box_xyxy"]) - chosen_center) <= radius
                for p in probes)
            picked["hold_probe_count"] = len(probes)
            picked["hold_support"] = int(support)
            picked["hold_probe_frames"] = [p["img_id"] for p in probes]
            # Agreement is evidence for stability, not identity. It may reject
            # an apparently strong but inconsistent winner, but it must never
            # promote a weak single-frame margin: a stable gripper fragment is
            # still the wrong object.
            if picked["status"] == "accepted" and support < len(probes):
                picked["status"] = "review_required"
                picked["reason"] = "inconsistent_hold_probes"
        record.update(picked)
        record["img_id"] = chosen_probe["img_id"]
        record["proposal_cache_path"] = chosen_probe["cache_path"]
        record["candidate_count"] = len(chosen_probe["ranked"])
        record["top_candidates"] = [
            {"bbox_xyxy": list(p.bbox), "mask_px": p.area, **feat}
            for p, feat in chosen_probe["ranked"][:5]
        ]
        results.append(record)

        vis = chosen_probe["image"].copy()
        h, w = vis.shape[:2]
        pts = np.array([[round(x * (w - 1)), round(y * (h - 1))]
                        for x, y in profile["regions"][role]["polygon"]], np.int32)
        cv2.polylines(vis, [pts], True, (255, 255, 255), 2)
        if "seed_box_xyxy" in picked:
            x0, y0, x1, y1 = map(int, picked["seed_box_xyxy"])
            color = ROLE_COLORS[role] if picked["status"] == "accepted" else (0, 0, 255)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 3)
        cv2.putText(vis, f"{role} c{cycle_idx}: {picked['status']}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        panels.append(vis)

    review = [r for r in results if r["status"] != "accepted"]
    manual = any(not r.get("automatic", True) for r in results)
    status = "review_required" if review or not results else ("manual" if manual else "accepted")
    report = {
        "schema_version": "1.0", "trial": args.trial, "status": status,
        "automatic": not manual, "event_summary": event_summary,
        "proposal_provenance": proposal_config,
        "events": events, "selections": results,
        "policy": {"low_confidence": "abstain", "production_sidecar_allowed": status == "accepted"},
    }
    report_path = os.path.join(args.out, "selection_report.json")
    write_json_safely(report_path, report)

    csv_path = os.path.join(args.out, "selections.csv")
    csv_tmp = csv_path + ".tmp"
    with open(csv_tmp, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["role", "cycle_idx", "event", "img_id", "status",
                         "automatic", "score", "confidence_margin",
                         "seed_x", "seed_y", "bbox_x0", "bbox_y0",
                         "bbox_x1", "bbox_y1"])
        for r in results:
            pt, box = r.get("seed_point_xy", [None, None]), r.get("seed_box_xyxy", [None] * 4)
            writer.writerow([r["role"], r["cycle_idx"], r["event"], r["img_id"],
                             r["status"], r.get("automatic", True), r.get("score"),
                             r.get("confidence_margin"), *pt, *box])
    backup_if_exists(csv_path)
    os.replace(csv_tmp, csv_path)

    if panels:
        thumb_h = 360
        thumbs = [cv2.resize(p, (round(p.shape[1] * thumb_h / p.shape[0]), thumb_h))
                  for p in panels]
        review_path = os.path.join(args.out, "selection_review.png")
        review_tmp = review_path + ".tmp.png"
        if not cv2.imwrite(review_tmp, np.hstack(thumbs)):
            raise RuntimeError(f"failed to write review image: {review_tmp}")
        backup_if_exists(review_path)
        os.replace(review_tmp, review_path)
    print(f"[write] {report_path} status={status} selections={len(results)}", flush=True)
    return 0 if status in {"accepted", "manual"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
