"""Decompose the selector's abstentions into correct ones and missed ones.

WHY
---
The headline number for the grasped role is "0/5 accepted", which reads as
total failure. It is not: it measures coverage and says nothing about whether
declining was the right call on each case.

An abstention is CORRECT when no valid answer exists in the candidate pool -
the correct object is simply not there at IoU >= 0.50, so any acceptance would
have been a guess. It is a MISS when the answer was present and the selector
declined anyway.

Those two are opposite outcomes and the deliverable's own acceptance criteria
treat abstention as "missing coverage, never an error". They had never been
separated, so a safety property that actually holds was being reported as a
failure.

WHAT THIS DOES NOT CLAIM
------------------------
It shows the selector declined and that declining was right. It does NOT show
it declined BECAUSE it detected the absence - both readings fit the data, and
with 5 cases they cannot be separated. Do not quote this as evidence that
abstention is well-calibrated.

Note also what is NOT a valid abstention case here. The excluded cycles
(lfdws_t004 cycles 1/3/4, lfdws_t005 cycle 2) were checked by gripper width at
their probe frames: 13.7, 18.2, 16.3 and 14.4 mm respectively. An object is
held in every one. They are MISSING-REFERENCE cases, not empty-gripper cases,
and adding them as expected abstentions would assert the opposite of the truth.

Usage:
    .venv_analysis/bin/python Code/abstention_audit.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from event_utils import mask_from_overlay

BUNDLE = "figures/interaction_bakeoff/input"
MANIFEST = "config/evaluation_manifest.yaml"
OUT = "figures/abstention_audit.txt"
GRASPED_BGR = (0, 255, 0)
POSITIVE_IOU = 0.50
RGB_DIR = "zed_zed_node_rgb_color_rect_image_compressed"

TRIALS = {
    "lfdws_t001": "Data/lfdws_t001/lfdws_t001",
    "lfdws_t002_new": "Data/lfdws_t002_new",
    "lfdws_t002_labexport": "Data/lfdws_t002_labexport/lfdws_t002",
    "lfdws_t004": "Data/lfdws_t004",
    "lfdws_t005": "Data/lfdws_t005",
}
# Frozen evaluation, 2026-09-03: the selector accepted 0 of 5 grasped cases,
# i.e. it abstained on every one.
FROZEN_ACCEPTED = set()


def iou(a, b):
    u = int((a | b).sum())
    return float((a & b).sum()) / u if u else 0.0


def reference_mask(rec, trial, img_id):
    summary = rec.get("reference_sidecar")
    if not summary or not os.path.exists(summary):
        return None
    with open(summary) as fh:
        for r in csv.DictReader(fh):
            if r.get("role") != "grasped":
                continue
            if os.path.splitext(r["img_filename"])[0] != img_id:
                continue
            ov = r.get("overlay_path")
            src = os.path.join(trial, RGB_DIR, img_id + ".png")
            if ov and os.path.exists(ov) and os.path.exists(src):
                return mask_from_overlay(ov, src, GRASPED_BGR)
            return None
    return None


def main():
    man = {r["id"]: r for r in yaml.safe_load(open(MANIFEST))["recordings"]}
    bench = json.load(open(os.path.join(BUNDLE, "benchmark.json")))
    lines = []

    def out(s):
        print(s, flush=True)
        lines.append(s)

    out("ABSTENTION AUDIT - grasped role, frozen selector")
    out("An abstention is correct when the pool contains no valid answer.")
    out("")
    out(f"{'case':<40}{'ceiling':>9}{'answer?':>9}{'did':>11}  verdict")

    correct = missed = 0
    for c in [x for x in bench["cases"] if x["role"] == "grasped"]:
        rid, img_id = c["recording_id"], c["img_id"]
        trial = TRIALS.get(rid)
        rec = man.get(rid, {})
        ref = reference_mask(rec, trial, img_id) if trial else None
        if ref is None or not ref.any():
            out(f"{c['case_id']:<40}{'-':>9}{'-':>9}{'-':>11}  no reference, skipped")
            continue
        masks = np.load(os.path.join(BUNDLE, c["proposal_cache"]),
                        allow_pickle=True)["masks"]
        ceiling = max(iou(masks[i].astype(bool), ref)
                      for i in range(masks.shape[0]))
        winnable = ceiling >= POSITIVE_IOU
        accepted = c["case_id"] in FROZEN_ACCEPTED
        did = "accepted" if accepted else "abstained"
        if accepted:
            verdict = "accepted"
        elif winnable:
            verdict = "MISS - answer was there"
            missed += 1
        else:
            verdict = "CORRECT - no answer existed"
            correct += 1
        out(f"{c['case_id']:<40}{ceiling:>9.3f}"
            f"{('yes' if winnable else 'NO'):>9}{did:>11}  {verdict}")

    out("")
    out(f"correct abstentions: {correct}")
    out(f"missed opportunities: {missed}")
    out(f"wrong accepts:        0   (the selector accepted nothing)")
    out("")
    out("Reading: '0/5 accepted' measures COVERAGE. The safety property - "
        "never publish a confident wrong label - held on every case tested. "
        "The failure is coverage, not correctness.")
    out("")
    out("Caveat: this shows declining was right, not that the selector "
        "declined BECAUSE it detected the absence. Five cases cannot "
        "separate those.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
