"""Ask a VLM to point at the grasped object; map the point to a cached proposal.

WHY POINTING AND NOT SET-OF-MARK
--------------------------------
Set-of-Mark (arXiv:2310.11441) overlays numbered marks on candidate regions and
asks the model for a number. It is designed for tens of regions. Our pools hold
60-277 proposals per frame (277 on lfdws_t004 after the dense run), and 200+
numerals stamped on one 960x540 wrist-camera frame is unreadable to a model or
a person. Pointing avoids that entirely: the model returns one (x, y), and the
cached proposal containing that point is the selection. It also plays to what
Molmo/RoboPoint were explicitly trained to do.

PRE-REGISTERED, FIXED BEFORE ANY RESULT WAS SEEN (2026-10-05)
--------------------------------------------------------------
PROMPT: the exact string in PROMPT below. One prompt. No prompt-shopping. If a
backend needs a different phrasing to parse at all, record it in the output
JSON as prompt_variant and say so in the report; do not quietly swap it.

PASS MARKS, against the 5 grasped cases across 4 independent groups:
    5/5  works. Take it to the lab with the data request.
    4/5  meets the frozen stop gate (>=4/5 across >=3 groups). Same action.
    3/5  better than anything tried (best so far 2/5) but not conclusive.
    <=2/5 no better than the existing hand-written rule. Say so plainly.

Chance is ~1/87 per case on the frozen pool and ~1/208 on the dense pool, so
5/5 by luck is ~1e-10. A zero-shot model has nothing fitted, which is exactly
why a small-n result from it is honest evidence where our fitted rules were
not. That reasoning collapses the moment anyone tunes the prompt on the answers.

NO GROUND TRUTH ON THIS MACHINE, BY DESIGN
------------------------------------------
figures/interaction_bakeoff/input/ is stamped inference_only_no_reference_masks
and carries no reference masks. This script writes predictions only. Scoring
happens where the ground truth lives, via Code/score_vlm_grasped_bakeoff.py.
Do not copy reference masks onto the GPU host to "check as you go" - that is
how a zero-shot result stops being zero-shot.

Usage (on the GPU host):
    python Code/run_vlm_grasped_bakeoff.py --backend molmo2
    python Code/run_vlm_grasped_bakeoff.py --backend robopoint
    python Code/run_vlm_grasped_bakeoff.py --backend qwen25vl
    # dense pool instead of the frozen one (only if dense_cache was pulled):
    python Code/run_vlm_grasped_bakeoff.py --backend molmo2 --pool dense
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import cv2
import numpy as np

BUNDLE = "figures/interaction_bakeoff/input"
BENCH = os.path.join(BUNDLE, "benchmark.json")
DENSE_CACHE = "figures/proposal_dataset/dense_cache/{rec}/{img}.npz"
OUT_DIR = "figures/vlm_grasped_bakeoff"

# ---------------------------------------------------------------------------
# THE PROMPT. Fixed 2026-10-05 before any model was run. Do not edit to chase
# a better number; that re-creates the overfitting this experiment exists to
# avoid.
# ---------------------------------------------------------------------------
PROMPT = (
    "This image is from a camera mounted on a robot's wrist, looking out past "
    "its two-finger gripper. The gripper is currently holding one object. "
    "Point at the object that the gripper is holding. "
    "Point at the object itself, not at the gripper or the fingers."
)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def grasped_cases():
    bench = json.load(open(BENCH))
    return [c for c in bench["cases"] if c["role"] == "grasped"]


def load_pool(case, pool):
    """Return (masks, meta). Frozen pool ships with the bundle; dense pool is
    the 2026-10-05 regeneration and is only present if it was pulled."""
    if pool == "frozen":
        path = os.path.join(BUNDLE, case["proposal_cache"])
    else:
        path = DENSE_CACHE.format(rec=case["recording_id"], img=case["img_id"])
    if not os.path.exists(path):
        return None, None, path
    d = np.load(path, allow_pickle=True)
    return d["masks"], json.loads(str(d["meta"])), path


def parse_points(text, w, h):
    """Extract (x, y) candidates from a model's reply.

    Handles Molmo's <point x="12.3" y="45.6"> XML, bare "(x, y)" pairs, and
    normalised 0-1 or 0-100 coordinates. Returns pixel coords, best first.
    """
    pts = []
    for m in re.finditer(r'x\d*\s*=\s*"([\d.]+)"\s*y\d*\s*=\s*"([\d.]+)"', text):
        pts.append((float(m.group(1)), float(m.group(2))))
    if not pts:
        for m in re.finditer(r'[\(\[]\s*([\d.]+)\s*,\s*([\d.]+)\s*[\)\]]', text):
            pts.append((float(m.group(1)), float(m.group(2))))
    out = []
    for x, y in pts:
        if x <= 1.0 and y <= 1.0:          # normalised 0-1
            x, y = x * w, y * h
        elif x <= 100.0 and y <= 100.0:    # Molmo emits percentages
            x, y = x / 100.0 * w, y / 100.0 * h
        out.append((float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))))
    return out


def proposal_at_point(masks, meta, x, y):
    """Smallest cached proposal containing the point.

    Smallest, not first: SAM returns nested subpart/part/whole masks for the
    same spot, and the whole-scene mask contains every point. Taking the
    smallest containing region is the only choice that does not trivially
    collapse to the background blob. Falls back to nearest centroid when the
    point lands outside every proposal.
    """
    xi, yi = int(round(x)), int(round(y))
    hits = [i for i in range(masks.shape[0]) if masks[i][yi, xi]]
    if hits:
        areas = [int(masks[i].sum()) for i in hits]
        return hits[int(np.argmin(areas))], "contains"
    best, bd = None, 1e18
    for i in range(masks.shape[0]):
        ys, xs = np.nonzero(masks[i])
        if not len(xs):
            continue
        d = (xs.mean() - x) ** 2 + (ys.mean() - y) ** 2
        if d < bd:
            best, bd = i, d
    return best, "nearest_centroid"


# ----------------------------- backends ------------------------------------
def backend_molmo2(model_id):
    """Ai2 Molmo 2 / MolmoAct 2. Pointing-native; emits <point x= y=> XML."""
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True,
                                         torch_dtype="auto", device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True,
                                                 torch_dtype="auto",
                                                 device_map="auto")

    def run(image_bgr, prompt):
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        inputs = proc.process(images=[img], text=prompt)
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate_from_batch(
                inputs, dict(max_new_tokens=200, stop_strings=["<|endoftext|>"]),
                tokenizer=proc.tokenizer)
        gen = out[0, inputs["input_ids"].size(1):]
        return proc.tokenizer.decode(gen, skip_special_tokens=True)
    return run


def backend_qwen25vl(model_id):
    """Qwen2.5-VL / Qwen3-VL. General grounding; returns coords in text."""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto")

    def run(image_bgr, prompt):
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt +
             " Reply with only the pixel coordinates as (x, y)."}]}]
        text = proc.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=128)
        return proc.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                 skip_special_tokens=True)[0]
    return run


def backend_robopoint(model_id):
    """RoboPoint (CoRL 2024, arXiv:2406.10721). LLaVA-style; emits point lists.

    Needs its own repo on the path: github.com/wentaoyuan/RoboPoint
    """
    sys.path.insert(0, os.environ.get("ROBOPOINT_REPO", ".external/RoboPoint"))
    import torch
    from robopoint.mm_utils import get_model_name_from_path
    from robopoint.model.builder import load_pretrained_model
    tok, model, image_processor, _ = load_pretrained_model(
        model_id, None, get_model_name_from_path(model_id))

    def run(image_bgr, prompt):
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        px = image_processor.preprocess(img, return_tensors="pt")["pixel_values"]
        px = px.half().to(model.device)
        ids = tok(f"USER: <image>\n{prompt} ASSISTANT:",
                  return_tensors="pt").input_ids.to(model.device)
        with torch.inference_mode():
            out = model.generate(ids, images=px, max_new_tokens=128,
                                 do_sample=False)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return run


BACKENDS = {
    "molmo2":    (backend_molmo2,    "allenai/Molmo-7B-D-0924"),
    "molmoact2": (backend_molmo2,    "allenai/MolmoAct-7B-D-0812"),
    "qwen25vl":  (backend_qwen25vl,  "Qwen/Qwen2.5-VL-7B-Instruct"),
    "robopoint": (backend_robopoint, "wentao-yuan/robopoint-v1-vicuna-v1.5-13b"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    ap.add_argument("--model", default=None, help="override the HF model id")
    ap.add_argument("--pool", default="frozen", choices=["frozen", "dense"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fn, default_id = BACKENDS[args.backend]
    model_id = args.model or default_id
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = args.out or os.path.join(
        OUT_DIR, f"predictions_{args.backend}_{args.pool}.json")

    cases = grasped_cases()
    log(f"backend={args.backend}  model={model_id}  pool={args.pool}")
    log(f"cases={len(cases)}  (expect 5 grasped)")
    log(f"PROMPT: {PROMPT}")

    log("loading model (first run downloads weights) ...")
    run = fn(model_id)
    log("model ready")

    preds = []
    for i, case in enumerate(cases, 1):
        img_path = os.path.join(BUNDLE, case["image"])
        image = cv2.imread(img_path)
        if image is None:
            log(f"({i}/{len(cases)}) {case['case_id']} SKIP unreadable image")
            continue
        h, w = image.shape[:2]
        masks, meta, pool_path = load_pool(case, args.pool)
        if masks is None:
            log(f"({i}/{len(cases)}) {case['case_id']} SKIP no pool at {pool_path}")
            continue

        t0 = time.time()
        try:
            reply = run(image, PROMPT)
        except Exception as exc:  # noqa: BLE001 - record and continue
            log(f"({i}/{len(cases)}) {case['case_id']} ERROR {type(exc).__name__}: {exc}")
            preds.append(dict(case_id=case["case_id"], error=str(exc)))
            continue
        dt = time.time() - t0

        pts = parse_points(reply, w, h)
        rec = dict(case_id=case["case_id"], recording_id=case["recording_id"],
                   independent_group=case["independent_group"],
                   img_id=case["img_id"], pool=args.pool,
                   pool_path=pool_path, n_proposals=int(masks.shape[0]),
                   prompt=PROMPT, raw_reply=reply, points=pts,
                   seconds=round(dt, 1))
        if pts:
            idx, how = proposal_at_point(masks, meta, *pts[0])
            rec.update(selected_index=int(idx), selection_mode=how,
                       selected_bbox=[float(v) for v in meta[idx]["bbox"]],
                       selected_area=int(meta[idx]["area"]))
            log(f"({i}/{len(cases)}) {case['case_id']} point={pts[0]} "
                f"-> proposal {idx} ({how}, area {meta[idx]['area']}) [{dt:.1f}s]")
        else:
            rec.update(selected_index=None, selection_mode="no_point_parsed")
            log(f"({i}/{len(cases)}) {case['case_id']} NO POINT PARSED [{dt:.1f}s]")
            log(f"      raw reply: {reply[:300]}")
        preds.append(rec)

    payload = dict(
        schema_version="1.0", created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        backend=args.backend, model=model_id, pool=args.pool, prompt=PROMPT,
        note="predictions only; no ground truth on this host by design. "
             "Score with Code/score_vlm_grasped_bakeoff.py where references live.",
        predictions=preds)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=1)
    log("")
    log(f"WROTE {out_path}")
    log(f"  parsed a point on {sum(1 for p in preds if p.get('selected_index') is not None)}"
        f"/{len(preds)} cases")
    log("  Commit and push this file; score it on the machine with the "
        "reference masks. Do NOT self-score here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
