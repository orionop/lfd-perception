"""Regenerate SAM proposals densely on the grasped-role evaluation frames.

WHY
---
Code/harvest_proposal_dataset.py measured the proposal-pool ceiling for the
GRASPED role for the first time and found it is the binding constraint:

    group            frames   pool has object   median best IoU
    t001_original         6               2/6             0.411
    t002_cube            12              2/12             0.257
    t004                 18             18/18             0.789
    t005                  5               5/5             0.972
    TOTAL                41             27/41

14 of 41 frames are unwinnable by ANY ranker because the correct object is not
in the pool at IoU >= 0.50. figures/contact_ceiling_study.json's 7/7 coverage
result, which is quoted in the docs as "proposal generation is not the limiting
stage", was measured for the CONTACT role only and does not hold for grasped.

t002_cube's 0.257 median is the multi-coloured-object failure already recorded
in CLAUDE.md (SAM segments sticker faces, not the cube), now quantified at the
pool level rather than the prompt level.

WHAT CHANGES
------------
The frozen cache was generated with deliberately conservative automask
settings. This run loosens exactly the parameters that govern small-object and
part-vs-whole recall, and nothing else:

    points_per_side          24  -> 48    denser sampling grid
    pred_iou_thresh        0.85  -> 0.70  keep lower-confidence masks
    stability_score_thresh 0.90  -> 0.80  keep less-stable masks
    min_mask_region_area    400  -> 200   keep smaller regions
    crop_n_layers             0  -> 1     re-run on image crops, which is the
                                          standard SAM remedy for small objects
                                          such as one held between fingers

This is a RECALL change, not a ranking change. It can only grow the candidate
pool. It is expected to add many wrong proposals too; that is acceptable here
because the question being asked is solely "is the correct object present at
all", which is the ceiling every ranker is bounded by.

HONESTY
-------
Writes to a NEW directory. The frozen cache under figures/deliverable_eval/ is
never touched, so the 27/41 baseline stays reproducible and this run cannot
retroactively improve any previously reported number.

Resumable: frames already written are skipped, so an interrupted run continues.

Usage:
    .venv_sam2/bin/python Code/regenerate_dense_proposals.py
    .venv_sam2/bin/python Code/regenerate_dense_proposals.py --limit 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from event_utils import mask_from_overlay

DATASET = "figures/proposal_dataset/grasped.csv"
MANIFEST = "config/evaluation_manifest.yaml"
OUT_DIR = "figures/proposal_dataset/dense_cache"
OUT_REPORT = "figures/proposal_dataset/dense_ceiling_report.txt"
CKPT = "sam_vit_h_4b8939.pth"
MODEL = "vit_h"
POSITIVE_IOU = 0.50
GRASPED_BGR = (0, 255, 0)

DENSE = dict(points_per_side=48, pred_iou_thresh=0.70,
             stability_score_thresh=0.80, min_mask_region_area=200,
             crop_n_layers=1)
FROZEN = dict(points_per_side=24, pred_iou_thresh=0.85,
              stability_score_thresh=0.90, min_mask_region_area=400,
              crop_n_layers=0)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def iou(a, b):
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else 0.0


def frames_from_dataset():
    """(recording, group, img_id) for every frame already harvested, plus the
    frozen pool's best IoU so the comparison is like-for-like."""
    best = {}
    with open(DATASET) as fh:
        for r in csv.DictReader(fh):
            k = (r["recording"], r["group"], r["img_id"])
            v = float(r["gt_iou"])
            if k not in best or v > best[k]:
                best[k] = v
    return sorted(best.items())


def trial_dirs():
    import yaml
    out = {}
    for rec in yaml.safe_load(open(MANIFEST))["recordings"]:
        out[rec["id"]] = (rec["trial"], rec["reference_sidecar"])
    return out


def overlay_index(summary_csv):
    idx = {}
    if not os.path.exists(summary_csv):
        return idx
    with open(summary_csv) as fh:
        for row in csv.DictReader(fh):
            if row.get("role") != "grasped":
                continue
            ov = row.get("overlay_path") or ""
            if ov and os.path.exists(ov):
                idx[os.path.splitext(row["img_filename"])[0]] = ov
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="only process the first N frames (smoke test)")
    args = ap.parse_args()

    if not os.path.exists(CKPT):
        raise SystemExit(f"missing {CKPT}")
    os.makedirs(OUT_DIR, exist_ok=True)

    frames = frames_from_dataset()
    if args.limit:
        frames = frames[:args.limit]
    log(f"frames to process: {len(frames)}")
    log(f"frozen settings: {FROZEN}")
    log(f"dense  settings: {DENSE}")

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    from segment_anything.utils.transforms import ResizeLongestSide

    # MPS does not support float64, and SAM's automatic mask generator builds
    # its point grid in float64 (np.linspace default) then multiplies it by an
    # int64 image-size array, so the product stays float64 and
    # torch.as_tensor(..., device="mps") raises:
    #   TypeError: Cannot convert a MPS Tensor to float64 dtype
    # Code/select_objects.py never hit this because it selects
    # "cuda" if torch.cuda.is_available() else "cpu" and so always ran on CPU.
    # Casting the transformed coordinates to float32 is the minimal fix and
    # does not change the sampled point locations at image resolution.
    if not getattr(ResizeLongestSide, "_float32_patched", False):
        _orig_apply_coords = ResizeLongestSide.apply_coords

        def _apply_coords_f32(self, coords, original_size):
            return _orig_apply_coords(self, coords, original_size).astype(np.float32)

        ResizeLongestSide.apply_coords = _apply_coords_f32
        ResizeLongestSide._float32_patched = True

    def build(device):
        log(f"loading SAM {MODEL} on {device} ...")
        sam = sam_model_registry[MODEL](checkpoint=CKPT).to(device)
        g = SamAutomaticMaskGenerator(sam, **DENSE)
        g.point_grids = [p.astype(np.float32) for p in g.point_grids]
        return g

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    gen = build(device)

    # Prove the device actually works on one real frame before committing to a
    # long unattended run; fall back to the CPU path that produced the frozen
    # cache rather than dying hours in.
    if device != "cpu":
        probe_rec, probe_group, probe_img = frames[0][0]
        probe_trial = trial_dirs()[probe_rec][0]
        probe_path = os.path.join(
            probe_trial, "zed_zed_node_rgb_color_rect_image_compressed",
            probe_img + ".png")
        probe = cv2.imread(probe_path)
        if probe is not None:
            try:
                t0 = time.time()
                gen.generate(cv2.cvtColor(probe, cv2.COLOR_BGR2RGB))
                log(f"device probe OK on {device} ({time.time() - t0:.1f}s/frame)")
            except Exception as exc:  # noqa: BLE001 - any device fault falls back
                log(f"device probe FAILED on {device}: {type(exc).__name__}: {exc}")
                log("falling back to cpu (the path that produced the frozen cache)")
                device = "cpu"
                gen = build(device)
    log(f"SAM ready on {device}")

    trials = trial_dirs()
    ov_cache = {}
    results = []
    t_start = time.time()

    for i, ((rec_id, group, img_id), frozen_best) in enumerate(frames, 1):
        trial, summary = trials[rec_id]
        if rec_id not in ov_cache:
            ov_cache[rec_id] = overlay_index(summary)
        overlays = ov_cache[rec_id]

        rgb_dir = os.path.join(trial,
                               "zed_zed_node_rgb_color_rect_image_compressed")
        img_path = os.path.join(rgb_dir, img_id + ".png")
        out_npz = os.path.join(OUT_DIR, rec_id, img_id + ".npz")
        os.makedirs(os.path.dirname(out_npz), exist_ok=True)

        if os.path.exists(out_npz):
            d = np.load(out_npz, allow_pickle=True)
            masks = d["masks"]
            log(f"({i}/{len(frames)}) {rec_id} {img_id} CACHED "
                f"{masks.shape[0]} proposals")
        else:
            image = cv2.imread(img_path)
            if image is None:
                log(f"({i}/{len(frames)}) {rec_id} {img_id} SKIP no source frame")
                continue
            t0 = time.time()
            anns = gen.generate(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            dt = time.time() - t0
            masks = np.stack([a["segmentation"] for a in anns]).astype(np.uint8) \
                if anns else np.zeros((0,) + image.shape[:2], np.uint8)
            meta = [{"bbox": [float(a["bbox"][0]), float(a["bbox"][1]),
                              float(a["bbox"][0] + a["bbox"][2]),
                              float(a["bbox"][1] + a["bbox"][3])],
                     "area": int(a["area"]),
                     "predicted_iou": float(a["predicted_iou"]),
                     "stability": float(a["stability_score"])} for a in anns]
            np.savez_compressed(
                out_npz, masks=masks, meta=json.dumps(meta),
                provenance=json.dumps({"cache_version": 2, "model": MODEL,
                                       "checkpoint": CKPT, "settings": DENSE,
                                       "source": "regenerate_dense_proposals.py"}))
            log(f"({i}/{len(frames)}) {rec_id} {img_id} "
                f"{masks.shape[0]} proposals in {dt:.1f}s")

        # --- measure the new ceiling on this frame -----------------------
        ov = overlays.get(img_id)
        if not ov or not os.path.exists(img_path):
            log(f"      no reference mask, ceiling not measurable")
            continue
        gt = mask_from_overlay(ov, img_path, GRASPED_BGR)
        if gt is None or not gt.any():
            log(f"      empty reference mask, ceiling not measurable")
            continue
        dense_best = max((iou(masks[j].astype(bool), gt)
                          for j in range(masks.shape[0])), default=0.0)
        results.append((rec_id, group, img_id, frozen_best, dense_best))
        flag = ""
        if frozen_best < POSITIVE_IOU <= dense_best:
            flag = "  *** RECOVERED ***"
        elif dense_best < POSITIVE_IOU:
            flag = "  still unwinnable"
        log(f"      best IoU  frozen {frozen_best:.3f} -> dense "
            f"{dense_best:.3f}{flag}")

    # ---------------- summary ----------------
    lines = []

    def out(m):
        print(m, flush=True)
        lines.append(m)

    out("")
    out("=" * 72)
    out("PROPOSAL-POOL CEILING, GRASPED ROLE: frozen vs dense")
    out("=" * 72)
    by_group = {}
    for rec_id, group, _img, fb, db in results:
        by_group.setdefault(group, []).append((fb, db))
    out(f"  {'group':<16}{'frames':>7}{'frozen':>12}{'dense':>12}"
        f"{'med frozen':>13}{'med dense':>12}")
    tf = td = tn = 0
    for grp in sorted(by_group):
        v = by_group[grp]
        fh_ = sum(1 for f, _ in v if f >= POSITIVE_IOU)
        dh = sum(1 for _, d in v if d >= POSITIVE_IOU)
        tf += fh_
        td += dh
        tn += len(v)
        out(f"  {grp:<16}{len(v):>7}{f'{fh_}/{len(v)}':>12}{f'{dh}/{len(v)}':>12}"
            f"{np.median([f for f, _ in v]):>13.3f}"
            f"{np.median([d for _, d in v]):>12.3f}")
    out(f"  {'TOTAL':<16}{tn:>7}{f'{tf}/{tn}':>12}{f'{td}/{tn}':>12}")
    out("")
    recovered = sum(1 for _, _, _, f, d in results
                    if f < POSITIVE_IOU <= d)
    lost = sum(1 for _, _, _, f, d in results if d < POSITIVE_IOU <= f)
    out(f"  frames recovered by denser proposals: {recovered}")
    out(f"  frames lost:                          {lost}")
    out(f"  elapsed: {(time.time() - t_start) / 60.0:.1f} min")
    out("")
    if td > tf:
        out("  READ THIS AS: the pool ceiling moved. The ranker can now be "
            "re-fitted on a pool that actually contains the answer more often.")
        out("  It does NOT mean selection improved. Re-run "
            "Code/harvest_proposal_dataset.py against this cache, then "
            "Code/fit_proposal_ranker.py, to find that out.")
    else:
        out("  READ THIS AS: denser proposals did NOT move the ceiling. The "
            "grasped object is not recoverable by SAM automask on these frames "
            "at any density tried, and the limitation is upstream of ranking "
            "AND of these settings.")
    out("")
    out("  Standing caveat: still 4 independent groups. Nothing here is "
        "evidence of generalisation.")

    with open(OUT_REPORT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"wrote {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
