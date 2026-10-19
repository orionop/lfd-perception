"""Ask the VLM WHAT each task-relevant object is, not only where it is.

WHY THIS EXISTS
---------------
Every run so far asked the model to point. The sidecar therefore says
"grasped" and "contact_receiver" but never says *what* the object is. That was
a real gap: a downstream consumer gets a role-tagged mask with no identity, so
two recordings of the same carrot are not relatable to each other, and the
lab's own question - "after N demonstrations, what is this object" - cannot be
answered from our output at all.

Getting a name costs one extra prompt against the same image and the same
model. It needs no new infrastructure and no training.

WHAT IT ALSO UNLOCKS, AND WHAT IT DOES NOT
-------------------------------------------
`archive/negative_results/object_identity*.py` tried to give objects stable
identities ACROSS recordings using DINOv2 embeddings and clustering. It failed:
it fragments one object into several identities and merges different objects
into one, with zero clean configurations across a 6-threshold x 2-crop-style
sweep. A name is a different mechanism entirely - it is stable under lighting,
viewpoint and scene change in a way an appearance embedding is not.

This script does NOT establish cross-recording identity. It produces labels.
Whether those labels are consistent enough to match objects across recordings
is a separate question that needs recordings nobody has looked at.

PRE-REGISTERED, FIXED BEFORE THE FIRST RUN (2026-10-19)
--------------------------------------------------------
A label is semantic, so "is it correct" has to be decided in advance or it
degenerates into judging the answer after seeing it. ACCEPTABLE below lists
the strings that count as correct for each case, chosen from the scene
descriptions in CLAUDE.md's trial table and NOT from any model output.

Matching is case-insensitive substring containment in either direction, so
"a red rubik's cube" matches "rubik's cube" and "cube" matches "cube".

Scoring is descriptive. There is no pass gate here: this closes a capability
gap in the sidecar, it does not decide whether the pipeline works. The
grasped/contact selection gate in LAB_DELIVERABLE_A.md is unaffected.

Usage (GPU host):
    python Code/run_vlm_object_label.py --backend molmo2
    python Code/run_vlm_object_label.py --backend molmo2 --role grasped
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2

BUNDLE = "figures/interaction_bakeoff/input"
OUT_DIR = "figures/vlm_object_label"

# ---------------------------------------------------------------------------
# PROMPTS. Fixed 2026-10-19 before any model was run.
# ---------------------------------------------------------------------------
PROMPT_GRASPED = (
    "This image is from a camera mounted on a robot's wrist, looking out past "
    "its two-finger gripper. The gripper is currently holding one object. "
    "What is that object? Answer with the object's common name in two or "
    "three words, and nothing else."
)
PROMPT_CONTACT = (
    "This image is from a camera mounted on a robot's wrist, looking out past "
    "its two-finger gripper. The robot is pressing against one object in the "
    "scene. What is the object being pressed against? Answer with the "
    "object's common name in two or three words, and nothing else."
)

# ---------------------------------------------------------------------------
# ACCEPTABLE ANSWERS, from CLAUDE.md's trial table. Written before any run.
# Scenes: t001 carrot/cup; t001_depth plate/screwdriver/charger;
# t002_* Rubik's cube; t004,t005 fastener on a wall pegboard;
# t001_labexport latch/hinge.
# ---------------------------------------------------------------------------
ACCEPTABLE = {
    "lfdws_t001__grasped__c1":           ["carrot"],
    "lfdws_t001__contact_receiver__c1":  ["cup", "mug", "glass"],
    "lfdws_t001_depth__contact_receiver__c1": ["plate", "dish"],
    "lfdws_t001_depth__contact_receiver__c2": ["screwdriver", "tool"],
    "lfdws_t001_depth__contact_receiver__c3": ["charger", "adapter", "plug"],
    "lfdws_t001_depth__contact_receiver__c4": ["charger", "adapter", "plug"],
    "lfdws_t001_depth__contact_receiver__c5": ["charger", "adapter", "plug"],
    "lfdws_t001_labexport__contact_receiver__c1": ["latch", "hinge", "pegboard"],
    "lfdws_t002_new__grasped__c1":       ["rubik", "cube"],
    "lfdws_t002_labexport__grasped__c1": ["rubik", "cube"],
    "lfdws_t004__grasped__c2":           ["nut", "fastener", "bolt", "screw"],
    "lfdws_t005__grasped__c1":           ["nut", "fastener", "bolt", "screw"],
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def label_matches(reply, accepted):
    r = reply.strip().lower()
    for a in accepted:
        if a.lower() in r:
            return True, a
    return False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="molmo2")
    ap.add_argument("--model", default=None)
    ap.add_argument("--role", default="all",
                    choices=["all", "grasped", "contact_receiver"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Reuse the backend loaders and case list from the pointing runner so there
    # is exactly one place where each model is wired up.
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_vlm_grasped_bakeoff import BACKENDS

    fn, default_id = BACKENDS[args.backend]
    model_id = args.model or default_id
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = args.out or os.path.join(
        OUT_DIR, f"labels_{args.backend}.json")

    bench = json.load(open(os.path.join(BUNDLE, "benchmark.json")))
    cases = [c for c in bench["cases"]
             if args.role == "all" or c["role"] == args.role]

    log(f"backend={args.backend}  model={model_id}  cases={len(cases)}")
    log(f"PROMPT_GRASPED: {PROMPT_GRASPED}")
    log(f"PROMPT_CONTACT: {PROMPT_CONTACT}")
    log("loading model ...")
    run = fn(model_id)
    log("model ready")

    rows, n_ok, n_scored = [], 0, 0
    for i, case in enumerate(cases, 1):
        cid = case["case_id"]
        img = cv2.imread(os.path.join(BUNDLE, case["image"]))
        if img is None:
            log(f"({i}/{len(cases)}) {cid} SKIP unreadable image")
            continue
        prompt = PROMPT_GRASPED if case["role"] == "grasped" else PROMPT_CONTACT
        t0 = time.time()
        try:
            reply = run(img, prompt)
        except Exception as exc:  # noqa: BLE001
            log(f"({i}/{len(cases)}) {cid} ERROR {type(exc).__name__}: {exc}")
            rows.append(dict(case_id=cid, role=case["role"], error=str(exc)))
            continue
        dt = time.time() - t0

        accepted = ACCEPTABLE.get(cid, [])
        ok, hit = label_matches(reply, accepted) if accepted else (None, None)
        if accepted:
            n_scored += 1
            n_ok += bool(ok)
        rows.append(dict(case_id=cid, role=case["role"],
                         recording_id=case["recording_id"],
                         independent_group=case["independent_group"],
                         prompt=prompt, raw_reply=reply,
                         acceptable=accepted, matched=hit,
                         correct=ok, seconds=round(dt, 1)))
        verdict = "-" if ok is None else ("CORRECT" if ok else "wrong")
        log(f"({i}/{len(cases)}) {cid:<42} -> {reply.strip()[:60]!r}  "
            f"[{verdict}]  {dt:.1f}s")

    payload = dict(schema_version="1.0",
                   created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   backend=args.backend, model=model_id,
                   prompt_grasped=PROMPT_GRASPED,
                   prompt_contact=PROMPT_CONTACT,
                   note="Object naming. Acceptable answers pre-registered "
                        "from CLAUDE.md's trial table before any run. "
                        "Descriptive: no pass gate, does not affect the "
                        "selection gate in LAB_DELIVERABLE_A.md.",
                   labels=rows)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=1)

    log("")
    log("=" * 70)
    log(f"LABELS CORRECT: {n_ok}/{n_scored}")
    by_role = {}
    for r in rows:
        if r.get("correct") is not None:
            b = by_role.setdefault(r["role"], [0, 0])
            b[0] += bool(r["correct"])
            b[1] += 1
    for role, (ok, tot) in sorted(by_role.items()):
        log(f"  {role:<18} {ok}/{tot}")
    log("")
    log("Reads as a capability check, not a validation: it says whether the "
        "model can name these objects, not whether names are consistent "
        "enough to match the same object across recordings. That needs "
        "recordings nobody has looked at.")
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
