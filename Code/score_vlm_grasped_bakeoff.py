"""Score VLM grasped-role predictions against the reference masks.

Runs on the machine that HAS the reference tracks (figures/*/identify/, which
are gitignored and never leave it). The GPU host only ever produces
predictions; see Code/run_vlm_grasped_bakeoff.py for why that split exists.

Pre-registered marks, fixed 2026-10-05 before any model was run:
    5/5   works. Take it to the lab with the data request.
    4/5   meets the frozen stop gate (>=4/5 across >=3 groups). Same action.
    3/5   better than anything tried (best so far 2/5), not conclusive.
    <=2/5 no better than the existing hand-written rule.

A selection counts correct at IoU >= 0.50 against the reference mask, matching
config/evaluation_manifest.yaml's selection_iou_threshold.

Usage:
    .venv_analysis/bin/python Code/score_vlm_grasped_bakeoff.py \
        figures/vlm_grasped_bakeoff/predictions_molmo2_frozen.json
"""
from __future__ import annotations

import csv
import json
import os
import sys

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from event_utils import mask_from_overlay

MANIFEST = "config/evaluation_manifest.yaml"
BUNDLE = "figures/interaction_bakeoff/input"
DENSE_CACHE = "figures/proposal_dataset/dense_cache/{rec}/{img}.npz"
GRASPED_BGR = (0, 255, 0)
POSITIVE_IOU = 0.50


def iou(a, b):
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def reference_mask(rec_id, img_id):
    """Recover the grasped reference mask for one frame, or None."""
    man = yaml.safe_load(open(MANIFEST))
    rec = next((r for r in man["recordings"] if r["id"] == rec_id), None)
    if rec is None:
        return None
    summary, trial = rec["reference_sidecar"], rec["trial"]
    if not os.path.exists(summary):
        return None
    overlay = None
    with open(summary) as fh:
        for row in csv.DictReader(fh):
            if row.get("role") == "grasped" and \
               os.path.splitext(row["img_filename"])[0] == img_id:
                overlay = row.get("overlay_path")
                break
    if not overlay or not os.path.exists(overlay):
        return None
    src = os.path.join(trial, "zed_zed_node_rgb_color_rect_image_compressed",
                       img_id + ".png")
    if not os.path.exists(src):
        return None
    return mask_from_overlay(overlay, src, GRASPED_BGR)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    path = sys.argv[1]
    payload = json.load(open(path))
    preds = payload["predictions"]

    print(f"predictions: {path}")
    print(f"backend={payload['backend']}  model={payload['model']}  "
          f"pool={payload['pool']}")
    print(f"prompt: {payload['prompt']}")
    print()

    rows, by_group = [], {}
    for p in preds:
        cid, rec_id, img_id = p["case_id"], p.get("recording_id"), p.get("img_id")
        grp = p.get("independent_group", "?")
        if p.get("selected_index") is None:
            rows.append((cid, grp, None, "no prediction"))
            by_group.setdefault(grp, []).append(0)
            continue
        ref = reference_mask(rec_id, img_id)
        if ref is None:
            rows.append((cid, grp, None, "NO REFERENCE - cannot score"))
            continue
        cache = (os.path.join(BUNDLE, f"proposals/{cid}.npz")
                 if p["pool"] == "frozen"
                 else DENSE_CACHE.format(rec=rec_id, img=img_id))
        if not os.path.exists(cache):
            rows.append((cid, grp, None, f"missing pool {cache}"))
            continue
        masks = np.load(cache, allow_pickle=True)["masks"]
        sel = masks[p["selected_index"]].astype(bool)
        if sel.shape != ref.shape:
            rows.append((cid, grp, None, "shape mismatch"))
            continue
        v = iou(sel, ref)
        ok = v >= POSITIVE_IOU
        rows.append((cid, grp, v, "CORRECT" if ok else "wrong"))
        by_group.setdefault(grp, []).append(int(ok))

    print(f"{'case':<40}{'group':<16}{'IoU':>7}  verdict")
    n_ok = 0
    for cid, grp, v, verdict in rows:
        n_ok += verdict == "CORRECT"
        print(f"{cid:<40}{grp:<16}{(f'{v:.3f}' if v is not None else '  -  '):>7}"
              f"  {verdict}")

    scored = sum(1 for r in rows if r[2] is not None)
    groups_ok = sum(1 for g, v in by_group.items() if any(v))
    print()
    print(f"CORRECT: {n_ok}/{len(rows)}   (scored {scored}; "
          f"{len(by_group)} groups, {groups_ok} with >=1 correct)")

    if n_ok >= 5:
        verdict = "PASS 5/5 - works. Take it to the lab with the data request."
    elif n_ok == 4:
        verdict = "PASS 4/5 - meets the frozen stop gate. Same action."
    elif n_ok == 3:
        verdict = ("PARTIAL 3/5 - beats everything tried (best so far 2/5) "
                   "but is not conclusive.")
    else:
        verdict = (f"FAIL {n_ok}/5 - no better than the existing hand-written "
                   "rule (2/5).")
    print(f"VERDICT: {verdict}")
    print()
    print("Standing caveat: 4 independent groups. Even 5/5 shows the method "
          "works on THIS data; it is not evidence of generalisation to new "
          "objects or scenes. That is what the extra recordings are for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
