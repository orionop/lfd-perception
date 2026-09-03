# Lab deliverable A — authoritative status

Last updated: 2026-09-03 (ground truth, feasibility ceiling, and early-hold scoring corrected)

This file is the single source of truth for the lab deliverable. Historical
notes in `calibration.yaml`, experimental scripts, figures, and the manuscript
must not override it.

## Scope

Deliverable A is an offline vision module that takes a lab-exported robot
demonstration and produces role-tagged object tracks in a JSON sidecar for the
downstream ROS 2 learning-from-demonstration pipeline.

The manuscript is workstream B. B is paused until A meets every acceptance
criterion below. Manuscript novelty is not an acceptance criterion for A.

## Current status

Execution is at a clean diagnostic checkpoint. The next selector work is
bounded by the feasibility result below; calibration and manuscript work are
not part of the current iteration.

The event-detection, SAM 2 propagation, multi-object aggregation, and sidecar
generation stages work on existing recordings. A is not complete yet because
object initialization is not reliable on unseen recordings and the sidecar has
not been demonstrated inside the lab's downstream ROS 2 consumer.

Calibration status: camera intrinsics are trusted. The fixed extrinsic that
maps camera-frame points into the `current_pose` subject frame is not trusted.
The matrix retained in
`calibration.yaml` is an experimental candidate only; production geometry is
disabled with `bota_to_camera.filled: false`.

Local implementation update (2026-09-03): the canonical multi-cycle event
detector, calibration-free proposal selector, safe-abstention contract,
one-command runner, versioned sidecar schema, reference consumer, and quality
gate are implemented. Unit and cached-proposal integration tests pass. The
selector has not yet passed the frozen 90% precision / 75% coverage bar on the
recording-group-held-out evaluation, so A remains incomplete and no automatic
performance claim is made. A reduced-density real-model smoke test on
`lfdws_t001` exercised the complete proposal path: it accepted the contact
receiver with 0.982 IoU against the reference track and safely abstained on the
grasped object. This is useful execution evidence, not a held-out result and
not an acceptance claim. The versioned sidecar was also rebuilt from the real
497-frame tracks (744 object records), passed the local validator with zero
errors or warnings, and was read by the reference consumer. The
calibration-independent regression suite passes. Actual lab ROS 2 consumption
remains an external acceptance gate. The full-density local selection sweep
(`figures/deliverable_eval/selection_evaluation.json`) currently reports:

- grasped: 0/5 automatic, 0.00 coverage (four independent groups available);
- contact: 2/7 automatic, 1.00 precision, 0.286 coverage (three independent
  groups available).

The contact ranking was corrected from containment to mask–region IoU, and the
grasp seed frame was moved to 20% of the closed hold so the scored instant
represents carrying rather than later placement. These corrections improved
the validity and breadth of the evaluation, but did not meet the frozen
coverage bar.

## Evaluation ground truth was defective; corrected 2026-09-03

The previously reported figures above were measured through a broken evaluator
and must not be quoted. `Code/evaluate_selection.py:reference_mask()` matched
ground truth by role *category*, collapsing `contact_receiver`, `tool_contact`
and `charger_contact` into one bucket and taking whichever overlay recovered
first. On `lfdws_t001_depth`, whose frames hold two or three contact objects,
four of the seven contact cases were therefore scored against an object the
robot was not touching. Re-scoring every cached proposal against the correct
reference shows the selector had in fact ranked the contacted object at or near
the top in all four:

| cycle | scored as | actual best candidate |
|---|---|---|
| c2 | 0.000 vs plate | 0.981 vs tool (rank 2) |
| c3 | 0.000 vs plate | 0.792 vs charger (rank 1) |
| c4 | 0.000 vs plate | 0.893 vs charger (rank 2) |
| c5 | 0.000 vs plate | 0.929 vs charger (rank 5) |

The manifest is now schema version 2 and declares ground truth **per cycle**.
Two authoring rules, both independent of the selector's own scores: contact
cycles take the object nearest the camera at the press frame, read from real
per-pixel depth inside each reference mask and cross-checked against visual
inspection of the event frame; grasped cycles require the reference mask to be
the object held in the gripper at the frame the selector is scored on, verified
by inspecting that frame.

`lfdws_t005` is admitted as grasped ground truth (cycle 1 only — its track ends
before cycle 2). `lfdws_t004` cycle 2 is also admitted at the early-hold scoring
frame: the nut is visibly between the fingers there, before it is threaded onto
the pegboard later in the same closed interval. Together these corrections
bring grasped to **5 cases across 4 independent recording groups**. `t004`
cycles 1, 3, and 4 remain excluded because their reference masks do not show a
carried object at the selector frame.

`Code/verify_grasped_reference.py` was written to supply eye-in-hand drift
evidence (a carried object holds a near-constant pixel while the scene sweeps
past). It confirms `lfdws_t001` at ratio 0.02 and `lfdws_t005` at 0.07 against
`lfdws_t004`'s whole-hold ratios of 0.59–1.40. It is **supporting evidence, not
the admission gate**:
it can only decide a cycle whose held phase overlaps real camera motion, and on
`lfdws_t002_*` it does not, so that recording reads `not_carried` despite a
reference track matching the object at IoU 0.986. Its output is now safely and
atomically persisted in `figures/grasped_reference_verification.json`.

## Current measured state (corrected ground truth, 2026-09-03)

- grasped: 0/5 automatic, coverage 0.00, **4 independent groups**, evidence
  sufficient;
- contact: 2/7 automatic, precision 1.00, coverage 0.286, 3 independent groups.

Coverage did not move when the ground truth was corrected, and that is the
expected result: those cycles were gate abstentions, so the defect had corrupted
*correctness*, not *acceptance*. It mattered nonetheless — had the acceptance
gate been repaired first, the four recovered cycles would have scored 0.0 IoU
against phantom ground truth and precision would have collapsed to 0.33, which
would have been read as a selector regression.

A deduplicated confidence margin is now in place: candidates overlapping the
winner above `duplicate_suppression_iou` (0.5, frozen in
`config/deliverable_rig.yaml`) are suppressed before the margin is measured, so
the margin reports separation from a genuinely different object rather than
SAM's own coarse and fine views of one object. This widened the worst margins
(`t001_depth` c2 0.001 → 0.032, `t005` c2 0.003 → 0.082) but changed no
acceptance decision.

## Why the selector still abstains

Every abstention is `low_score_or_margin`, and in every case the *score* clears
`min_score`; only the margin fails. Feature diagnostics on the training folds
(`lfdws_t001`, `lfdws_t001_depth`) show why the score band is only ~0.03 wide:

- `region_proximity` is 1.000 for effectively every candidate, yet carries 0.40
  of the contact score;
- `appearance_novelty` is 0.000 and `temporal_stability` 1.000 throughout;
- `stationarity` sits at its 0.5 "unknown" default because the `bg_flow > 0.25`
  gate rarely fires.

So roughly two thirds of the contact score is information-free, and
`region_overlap` is doing all the discriminating. That feature has a scale bias:
on cycles 4 and 5 a larger wrong mask (area fraction 0.151 / 0.290) outranks the
correct object (0.079 / 0.086). A size penalty would fix both and break cycle 1,
where the correct object is a plate at area fraction 0.309 — the same failure
`auto_seed.py`'s area cap already demonstrated. The next iteration must replace
the saturated features with ones that vary, not reweight the existing set.

That is now a measured stop condition, not intuition. The cached-proposal
feasibility study found the correct contact object in the candidate pool in
**7/7** cases, so proposal generation is adequate on this evaluation set. The
eight-feature baseline (`figures/contact_ceiling_existing_features.json`) tested 6,561
normalized signed linear rules: recording-group-separated selection achieved
1 correct acceptance out of 2 (precision 0.50, coverage 0.286), while even the
deliberately optimistic all-data fit accepted only 2/7.

A bounded ninth cue, local optical-flow contrast across the force-contact
instant, was then tested in 19,683 rules
(`figures/contact_flow_contrast_study.json`). It improved the optimistic all-data fit
to 3/7, but group-separated precision collapsed to 0.40 (2 correct among 5
accepted). It is therefore rejected from production. These are practical stop
tests, not proofs against every nonlinear method. Their actionable conclusion
is narrow: **keep the proposal pool, reject local flow contrast, and seek a
different contact-identity cue; do not continue tuning these weights.**

Open design question, deliberately not decided unilaterally: real depth
separates all five `t001_depth` contact cycles cleanly, but it was also used to
author their ground truth, so adopting it as a selector feature would make those
cycles partly self-confirming and would only apply to depth-bearing recordings.

The first safe multi-frame implementation probes grasped cycles at 20%, 27.5%,
and 35% of the closed hold and records spatial support. The event target is 20%,
early enough to recover `t004` cycle 2 before placement, although the current
production report still records whichever probe has the highest legacy score.
Consensus cannot promote an ambiguous identity, so the aggregate production
result remains unchanged.

The planned attachment-transition feasibility gate was executed before any
production integration (`figures/grasp_attachment_study.json`). Anchoring at the
frozen 20% frame exposed a more fundamental limitation: a proposal or local
union reaches IoU 0.5 in only **3/5** cases. The two `t002` exports reach only
0.266 and 0.250 because the object mask is still incomplete at that instant.
Across seven attachment features and 2,186 signed rules, even the deliberately
optimistic all-data fit accepted **0/5**. The gate therefore failed and no
attachment score was added to production. Per the frozen plan, contact-endpoint
engineering was not started.

Local fragmented-proposal unions are also implemented as candidates. On the
`t001` diagnostic frame, one union reaches 0.568 IoU against the reference,
but it is not the top-scoring candidate under the frozen rig features. It is
therefore retained for the next identity-scoring iteration and is not promoted
through a lowered threshold.

## Acceptance criteria

A is complete only when all of the following are demonstrated:

1. Automatic object selection
   - Select the correct `grasped` and `contact_receiver` objects without a
     recording-specific pixel or manually supplied prompt.
   - Freeze the method before evaluating it on recordings excluded from its
     development.
   - Report seed-inside-mask success and end-to-end track quality, including
     failures. The old `auto_seed.py` result (1/6) and fitted constant pixels
     are baselines, not accepted solutions.

2. Trusted geometric grounding
   - Recover the fixed transform that maps camera-frame points into the
     `current_pose` subject frame through an explicit calibration procedure,
     preferably the existing ChArUco hand-eye workflow.
   - Record the precise transform direction, frame names, units, and
     `current_pose` subject frame.
   - Validate with calibration reprojection error and independent physical or
     image points before enabling `bota_to_camera.filled`.
   - Evaluate wrench-projected contact seeds on held-out demonstrations.

3. Downstream integration
   - Freeze and document the `objects.json` schema.
   - Run a generated sidecar through the lab's actual downstream ROS 2/LfD
     consumer, not only the local builder.
   - Preserve the command, environment, input recording, output, and pass/fail
     evidence as a reproducible integration test.

4. Reproducible handover
   - One documented canonical command path processes a new export.
   - Representative sensor combinations are regression-tested.
   - Setup, checkpoints, expected outputs, manual interventions, and known
     limitations are documented.

## Current stop and next authorized decision

The bounded local selector plan reached its stop condition. Production remains
at grasped 0/5 and contact 2/7; no failed feasibility cue was promoted and there
are no new accepted cases to propagate. Do not continue feature or weight
tuning automatically.

The next work must begin with one explicit change of evidence, not another
ranking variation: either revise the grasp proposal timing and re-freeze the
evaluation before inspecting results, obtain diverse unseen recordings, or
recover trusted geometric contact grounding. Actual downstream ROS 2
integration remains an external final gate.

## In progress: bounded external-model compatibility test (2026-09-03 → )

Chosen change of evidence: test whether an external hand/interaction-object
detector (HOI-DETR) or a motion-foreground segmenter (DistinctNet) can supply
a ranking/seeding signal the frozen eight-feature rig cannot, on the same 5
grasp / 7 contact cases, without touching the proposal pool or the evaluator.
Full plan: `Docs/MONDAY_INTERACTION_BAKEOFF.md`. This is bounded, stop-gated,
and does not re-open weight tuning on the existing features.

Preparation complete, execution blocked only on RTX 4080 host access
(available 2026-09-07):

- Benchmark bundle frozen: `figures/interaction_bakeoff/input/` (5 grasp + 7
  contact cases, event images, DistinctNet raw/stabilized frame pairs, cached
  SAM proposal `.npz`, no reference masks — leak-safe by construction).
- Adapters (`Code/run_hoi_detr_bakeoff.py`, `Code/run_distinctnet_bakeoff.py`,
  `Code/score_interaction_bakeoff.py`, `Code/export_interaction_bakeoff.py`)
  written and unit-tested (`tests/test_interaction_bakeoff.py`), then verified
  by hand against the real, freshly-cloned upstream repos at the pinned
  commits (module globals, `predictions.json` schema, `Predictor` signature,
  checkpoint URLs) — not run, since neither model has a compatible GPU here.
- Both repos pre-cloned and pinned in `.external/interaction_bakeoff/`
  (gitignored staging dir, not pushed); `scripts/run_interaction_bakeoff_gpu.sh`
  will find them already at the correct commit and skip re-cloning.
- Checkpoint downloads (HOI-DETR 5.85 GB, DistinctNet smaller) deliberately
  **not** pre-staged — this machine had 13 GB free on a 97%-full disk; the
  script downloads both automatically on the GPU host on first run.
- HOI-DETR safeguard: a contact prediction counts only with a complete
  gripper/hand → 1st-object → 2nd-object relation chain; DistinctNet safeguard:
  a foreground mask counts only above 0.50 IoU against a cached SAM proposal.
  Neither can pass by construction alone.

**Known domain-mismatch risk, going in eyes open:** HOI-DETR is trained and
evaluated exclusively on human-hand datasets (Hands23, FineBio, HOIST,
HD-EPIC) — its "hand" class has never seen a robot gripper. The Franka gripper
looks nothing like a human hand, so its hand→object relation chain may fail to
form on most or all frames, which would abstain both grasp and contact
regardless of the object-detection quality. A clean 0/5 here should be read as
"human-hand detector didn't transfer to a robot gripper," a narrower and less
surprising conclusion than "interaction models don't help," not as a
refutation of the whole approach. DistinctNet's foreground/motion segmentation
is class-agnostic and has no such mismatch; it is the more informative signal
of the two, and is also the one that matches the eye-in-hand pose-cancellation
insight already established for `grasped_seed_pixel.py` (raw vs.
camera-stabilized frame pairs is a direct test of the same idea).

Stop gate (unchanged from the plan, reproduced here since this file is
authoritative): grasp passes only with ≥4/5 accepted, all correct, across
≥3 independent groups; contact passes only with ≥6/7 accepted, all correct,
across all 3 groups. If neither model passes grasp, stop model integration and
wait for new recordings. If contact fails, retain the 7/7 proposal-pool result
and wait for calibrated geometric evidence — do not resume heuristic-score
tuning. No result from this test authorizes production integration by itself;
only successful, stop-gate-passing evidence may be wired into the selector as
optional ranking/seeding input, per the plan's integration step.
