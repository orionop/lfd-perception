"""Use the VLM's point as a SAM 2 prompt instead of a lookup into a cached pool.

WHY
---
Round 2 (RESULTS.md, 2026-10-05) scored 0/5 under the pre-registered criterion,
but the decomposition showed the model was rarely at fault:

    case                  point on true object?  pool ceiling  what failed
    lfdws_t001            no                     0.568         model pointed wrong
    lfdws_t002_new        YES                    0.245         pool - unwinnable
    lfdws_t002_labexport  YES                    0.314         pool - unwinnable
    lfdws_t004            YES                    0.948         mapping code
    lfdws_t005            YES                    0.974         mapping code

Molmo located the held object in 4 of 5 cases. Two cases were impossible
because the cached pool's best mask is below the 0.50 bar. Two more were lost
by Code/run_vlm_grasped_bakeoff.py's rule of taking the SMALLEST cached
proposal containing the point: on t004 and t005 that grabbed a ~1.7k px
fragment while the correct ~4k px object mask, also containing the point, sat
in the same pool at IoU 0.95-0.97. That rule was chosen to avoid selecting the
whole-scene blob and overshot into the part-vs-whole failure instead.

Prompting SAM 2 with the point removes both failure modes at once: the mask is
generated fresh rather than looked up, so the pool ceiling does not apply, and
SAM 2 chooses the granularity rather than a size heuristic of ours.

PRE-REGISTERED, FIXED BEFORE THIS SCRIPT WAS FIRST RUN (2026-10-05)
-------------------------------------------------------------------
Stated in writing before execution because the Round 2 scores were already
known, which makes every later choice a potential post-hoc fit:

  INPUT     Molmo's points, exactly as produced on the GPU host. Frozen. This
            script never re-queries a VLM and never adjusts a point.
  RULE      One positive point prompt to SAM 2, multimask_output=True, and
            take the mask SAM 2 ranks highest BY ITS OWN CONFIDENCE. Not the
            one that scores best against the reference - that would be reading
            the answer. SAM 2's own ranking is its documented default.
  CRITERION IoU >= 0.50 against the reference mask, as everywhere else.
  CEILING   4/5. On lfdws_t001 the point is not on the object, so SAM 2 will
            faithfully segment the wrong thing and that case is lost by
            construction.
  MARKS     4/4 of the winnable cases (4/5 overall) -> meets the frozen stop
            gate, take to the lab with the data request.
            3/5 -> beats everything tried (best so far 2/5), not conclusive.
            <=2/5 -> no better than the existing hand-written rule.

ONE RUN. Whatever it prints is the result, including a bad one.

Usage:
    .venv_sam2/bin/python Code/score_vlm_sam2_prompt.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PRED = "figures/vlm_grasped_bakeoff/predictions_molmo2_frozen.json"
BUNDLE = "figures/interaction_bakeoff/input"
CKPT = "sam2.1_hiera_large.pt"
CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"
OUT = "figures/vlm_grasped_bakeoff/sam2_prompt_result.txt"
POSITIVE_IOU = 0.50


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    from score_vlm_grasped_bakeoff import reference_mask, iou

    preds = json.load(open(PRED))["predictions"]
    bench = {c["case_id"]: c for c in
             json.load(open(os.path.join(BUNDLE, "benchmark.json")))["cases"]}

    log("loading SAM 2 (torch import on this venv takes several minutes) ...")
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    predictor = SAM2ImagePredictor(build_sam2(CFG, CKPT, device=device))
    log(f"SAM 2 ready on {device}")

    lines, n_ok, rows = [], 0, []
    for p in preds:
        cid = p["case_id"]
        case = bench[cid]
        img = cv2.imread(os.path.join(BUNDLE, case["image"]))
        ref = reference_mask(p["recording_id"], p["img_id"])
        if img is None or ref is None:
            rows.append((cid, p["independent_group"], None, None, "no input/reference"))
            continue
        x, y = p["points"][0]

        predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        with torch.inference_mode():
            masks, scores, _ = predictor.predict(
                point_coords=np.array([[x, y]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=True)
        # SAM 2's own ranking. Pre-registered: not the best-against-reference.
        pick = int(np.argmax(scores))
        m = masks[pick].astype(bool)
        v = iou(m, ref)
        ok = v >= POSITIVE_IOU
        n_ok += ok
        # Reported for transparency only; never used to choose.
        all_ious = [round(float(iou(masks[i].astype(bool), ref)), 3)
                    for i in range(masks.shape[0])]
        rows.append((cid, p["independent_group"], v, float(scores[pick]),
                     "CORRECT" if ok else "wrong"))
        log(f"{cid:<38} pt=({x:.0f},{y:.0f}) sam2_conf={scores[pick]:.3f} "
            f"IoU={v:.3f} {'CORRECT' if ok else 'wrong'}   (all 3: {all_ious})")

    def out(s):
        print(s, flush=True)
        lines.append(s)

    out("")
    out("=" * 74)
    out("VLM POINT -> SAM 2 PROMPT, grasped role, 5 cases / 4 groups")
    out("=" * 74)
    out(f"{'case':<38}{'group':<16}{'IoU':>7}  verdict")
    for cid, grp, v, conf, verdict in rows:
        out(f"{cid:<38}{grp:<16}"
            f"{(f'{v:.3f}' if v is not None else '  -  '):>7}  {verdict}")
    out("")
    out(f"CORRECT: {n_ok}/{len(rows)}")
    out("")
    out("Comparison, same 5 cases:")
    out("  hand-written rule (best of everything tried)   2/5")
    out("  VLM point -> smallest cached proposal          0/5")
    out(f"  VLM point -> SAM 2 prompt (this run)           {n_ok}/5")
    out("")
    if n_ok >= 4:
        out("VERDICT: PASS - meets the frozen stop gate (>=4/5 across >=3 "
            "groups). Take to the lab with the data request.")
    elif n_ok == 3:
        out("VERDICT: PARTIAL 3/5 - beats everything tried, not conclusive.")
    else:
        out(f"VERDICT: FAIL {n_ok}/5 - no better than the hand-written rule.")
    out("")
    out("Ceiling was 4/5 by construction: on lfdws_t001 the point is not on "
        "the object, so that case was unwinnable here.")
    out("Standing caveat: 4 independent groups. This shows whether the method "
        "works on THIS data. It is not evidence of generalisation to new "
        "objects or scenes - that is what more recordings are for.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
