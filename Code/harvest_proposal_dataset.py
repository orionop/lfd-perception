"""Turn already-propagated reference tracks into a labelled proposal dataset.

WHY THIS EXISTS
---------------
Selection is the only broken stage of the deliverable: the correct object is in
the proposal pool (contact 7/7) and is not being picked (grasped 0/5, contact
2/7). Every attempt to fix it so far has hand-designed a rule against TWELVE
labelled cycles:

    6,561-rule search      best 1/2 group-separated, 2/7 all-data
    19,683-rule extension  3/7 all-data, group-separated precision 0.40
    attachment gate        3/5 held-out, 0/5 even when fitted
    stationarity           figures/motion_selection_held.csv, loses to random
    HOI-DETR / DistinctNet bakeoff, neither integrated

The all-data-beats-held-out gap in those numbers is the signature of fitting
noise, not of a bad hypothesis. Twelve points cannot separate a good rule from
a lucky one when each decision is 1-of-~87.

What was never used: the reference tracks in figures/*/identify/ are full SAM 2
propagations of the CORRECT object across every frame of each recording (5,781
rows on disk). They are consumed only as pipeline output. As supervision they
turn one seeded cycle into hundreds of labelled frames, which is exactly the
bootstrap that Grasp2Vec and arXiv:2305.06305 rely on (the latter trains its
grasp-segmentation model on 100-200 labelled images, then self-supervises
thousands). This script performs that harvest.

SCOPE: GRASPED ONLY, AND WHY
----------------------------
The grasped object is well defined for the whole closed-gripper hold, so every
frame inside the hold is a legitimate label for the same object.

Contact is NOT. The contacted object is only defined at the press instant, and
config/evaluation_manifest.yaml declares it per cycle for exactly that reason
(lfdws_t001_depth holds three contact objects in one frame and a different one
is touched in each cycle). Labelling frames around the press would manufacture
label noise and then measure it. Contact therefore stays at its 7 declared
cycles and is not harvested here.

WHAT THIS DOES NOT FIX
----------------------
This multiplies labelled FRAMES, not independent SCENES. The grasped role still
spans 4 independent recording groups, and frames drawn from one hold are
correlated. That is enough to FIT weights honestly; it is not enough to prove
generalisation. Validation power needs more recordings and nothing here
substitutes for that.

Output: figures/proposal_dataset/grasped.csv, one row per (frame, proposal).

Usage:
    .venv_analysis/bin/python Code/harvest_proposal_dataset.py
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from event_utils import mask_from_overlay

MANIFEST = "config/evaluation_manifest.yaml"
RIG = "config/deliverable_rig.yaml"
CACHE_GLOB = "figures/deliverable_eval/{rec}/proposal_cache/*.npz"
OUT_DIR = "figures/proposal_dataset"
OUT_CSV = os.path.join(OUT_DIR, "grasped.csv")

# Overlay colour of each reference role, as written by the propagation scripts.
COLORS = {
    "grasped": (0, 255, 0),
    "contact_receiver": (255, 0, 255),
    "tool_contact": (0, 165, 255),
    "charger_contact": (0, 215, 255),
}

# A proposal counts as the correct object at this overlap, matching
# config/evaluation_manifest.yaml's selection_iou_threshold.
POSITIVE_IOU = 0.50


def log(msg):
    print(msg, flush=True)


def iou(a, b):
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 0.0


def polygon_mask(shape, polygon):
    h, w = shape
    pts = np.array([[int(round(x * w)), int(round(y * h))]
                    for x, y in polygon], np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def border_sides(mask):
    sides = 0
    sides += bool(mask[0, :].any())
    sides += bool(mask[-1, :].any())
    sides += bool(mask[:, 0].any())
    sides += bool(mask[:, -1].any())
    return sides


def proximity_score(mask, region):
    """Centroid-distance score, matching Code/select_objects.py's intent."""
    if not mask.any() or not region.any():
        return 0.0
    ys, xs = np.nonzero(mask)
    ry, rx = np.nonzero(region)
    cy, cx = ys.mean(), xs.mean()
    ry, rx = ry.mean(), rx.mean()
    h, w = mask.shape
    d = np.hypot(cy - ry, cx - rx) / np.hypot(h, w)
    return float(np.exp(-4.0 * d))


def features_for(mask, meta, region, image):
    """Per-proposal features. No calibration, no extrinsic, single frame.

    Deliberately keeps predicted_iou and stability SEPARATE. Code/select_objects
    .py averages them into one `sam_quality` term before scoring, which throws
    away whatever independent signal either carries.
    """
    h, w = mask.shape
    area = int(mask.sum())
    if area == 0:
        return None
    x0, y0, x1, y1 = [float(v) for v in meta["bbox"]]
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    ys, xs = np.nonzero(mask)
    cy, cx = float(ys.mean()), float(xs.mean())

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image is not None else None
    if gray is not None:
        inside = gray[mask]
        outside = gray[~mask]
        contrast = float(abs(inside.mean() - outside.mean()) / 255.0)
        texture = float(inside.std() / 255.0)
    else:
        contrast = texture = 0.0

    return {
        "area_fraction": area / float(h * w),
        "log_area": float(np.log10(max(area, 1))),
        "fill_ratio": area / (bw * bh),
        "aspect": bw / bh,
        "centroid_x": cx / w,
        "centroid_y": cy / h,
        "dist_center": float(np.hypot(cy - h / 2.0, cx - w / 2.0) /
                             np.hypot(h / 2.0, w / 2.0)),
        "bbox_w_frac": bw / w,
        "bbox_h_frac": bh / h,
        "border_sides": float(border_sides(mask)),
        "predicted_iou": float(meta["predicted_iou"]),
        "stability": float(meta["stability"]),
        "region_overlap": iou(mask, region),
        "region_proximity": proximity_score(mask, region),
        "region_containment": float((mask & region).sum()) / max(1, area),
        "contrast": contrast,
        "texture": texture,
    }


def reference_masks_by_frame(summary_csv, role_filter):
    """frame img_id -> recovered reference mask, for one reference role."""
    out = {}
    if not os.path.exists(summary_csv):
        return out
    with open(summary_csv) as fh:
        for row in csv.DictReader(fh):
            if row.get("role") != role_filter:
                continue
            overlay = row.get("overlay_path") or ""
            if not overlay or not os.path.exists(overlay):
                continue
            img_id = os.path.splitext(row["img_filename"])[0]
            out[img_id] = overlay
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=CACHE_GLOB,
                    help="glob for proposal caches; {rec} is the recording id. "
                         "Default is the frozen deliverable_eval cache.")
    ap.add_argument("--out", default=OUT_CSV, help="output CSV path")
    args = ap.parse_args()
    cache_glob, out_csv = args.cache, args.out

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    log(f"cache glob: {cache_glob}")
    log(f"output:     {out_csv}")
    manifest = yaml.safe_load(open(MANIFEST))
    rig = yaml.safe_load(open(RIG))
    poly = rig["regions"]["grasped"]["polygon"]

    rows = []
    n_pos = n_neg = 0

    for rec in manifest["recordings"]:
        gts = [g for g in rec.get("ground_truth", [])
               if g["selector_role"] == "grasped"]
        if not gts:
            continue
        rec_id = rec["id"]
        group = rec["independent_group"]
        trial = rec["trial"]
        summary = rec["reference_sidecar"]

        overlays = reference_masks_by_frame(summary, "grasped")
        caches = sorted(glob.glob(cache_glob.format(rec=rec_id)))
        log(f"[{rec_id}] group={group} cached_frames={len(caches)} "
            f"reference_frames={len(overlays)}")
        if not overlays:
            log(f"[{rec_id}]   SKIP: no recoverable grasped reference track")
            continue

        rgb_dir = os.path.join(trial, "zed_zed_node_rgb_color_rect_image_compressed")

        for cache_path in caches:
            img_id = os.path.splitext(os.path.basename(cache_path))[0]
            if img_id not in overlays:
                log(f"[{rec_id}]   frame {img_id}: no reference mask, skipped")
                continue

            img_path = os.path.join(rgb_dir, img_id + ".png")
            if not os.path.exists(img_path):
                log(f"[{rec_id}]   frame {img_id}: source frame missing, skipped")
                continue
            image = cv2.imread(img_path)

            # mask_from_overlay differences the overlay against its own source
            # frame, so both paths are required.
            gt = mask_from_overlay(overlays[img_id], img_path, COLORS["grasped"])
            if gt is None or not gt.any():
                log(f"[{rec_id}]   frame {img_id}: empty reference mask, skipped")
                continue

            data = np.load(cache_path, allow_pickle=True)
            masks = data["masks"]
            meta = json.loads(str(data["meta"]))
            region = polygon_mask(masks.shape[1:], poly)

            hits = 0
            for idx in range(masks.shape[0]):
                m = masks[idx].astype(bool)
                if gt.shape != m.shape:
                    continue
                f = features_for(m, meta[idx], region, image)
                if f is None:
                    continue
                overlap = iou(m, gt)
                label = int(overlap >= POSITIVE_IOU)
                hits += label
                n_pos += label
                n_neg += 1 - label
                rows.append(dict(
                    recording=rec_id, group=group, img_id=img_id,
                    proposal_idx=idx, gt_iou=round(overlap, 4),
                    label=label, **{k: round(float(v), 6)
                                    for k, v in f.items()}))
            log(f"[{rec_id}]   frame {img_id}: {masks.shape[0]} proposals, "
                f"{hits} positive, best_iou={max((iou(masks[i].astype(bool), gt) for i in range(masks.shape[0])), default=0):.3f}")

    if not rows:
        log("NO ROWS HARVESTED - nothing written.")
        return 1

    cols = list(rows[0].keys())
    with open(out_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)

    groups = sorted({r["group"] for r in rows})
    frames = len({(r["recording"], r["img_id"]) for r in rows})

    # ---- proposal-pool ceiling, per group --------------------------------
    # figures/contact_ceiling_study.json established 7/7 pool coverage for the
    # CONTACT role and that result has been quoted as "proposal generation is
    # not the limiting stage". It was never measured for GRASPED. It is
    # measured here, because a ranker cannot select an object the pool does
    # not contain, and an unwinnable frame must not be charged to selection.
    per_frame = {}
    for r in rows:
        k = (r["recording"], r["img_id"])
        cur = per_frame.get(k)
        if cur is None or r["gt_iou"] > cur[0]:
            per_frame[k] = (r["gt_iou"], r["group"])
    by_group = {}
    for (rec_id, _img), (best, grp) in per_frame.items():
        by_group.setdefault(grp, []).append(best)

    log("")
    log("PROPOSAL-POOL CEILING FOR THE GRASPED ROLE")
    log(f"  {'group':<16}{'frames':>7}{'pool has object':>18}{'median best IoU':>18}")
    total_hit = total_n = 0
    for grp in sorted(by_group):
        vals = by_group[grp]
        hit = sum(1 for v in vals if v >= POSITIVE_IOU)
        total_hit += hit
        total_n += len(vals)
        log(f"  {grp:<16}{len(vals):>7}{f'{hit}/{len(vals)}':>18}"
            f"{float(np.median(vals)):>18.3f}")
    log(f"  {'TOTAL':<16}{total_n:>7}{f'{total_hit}/{total_n}':>18}")
    if total_hit < total_n:
        log(f"  {total_n - total_hit} of {total_n} frames are UNWINNABLE by any "
            "ranker: the object is not in the pool at IoU >= "
            f"{POSITIVE_IOU:.2f}.")

    log("")
    log(f"WROTE {out_csv}")
    log(f"  rows (proposal instances): {len(rows)}")
    log(f"  frames: {frames}")
    log(f"  independent groups: {len(groups)} {groups}")
    log(f"  positive: {n_pos}  negative: {n_neg}  "
        f"base rate: {100.0 * n_pos / max(1, len(rows)):.2f}%")
    log("")
    log("Reminder: this multiplies FRAMES, not SCENES. Still "
        f"{len(groups)} independent groups. Fit on it; do not claim "
        "generalisation from it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
