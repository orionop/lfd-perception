# Manuscript B — publication strategy

Last updated: 2026-09-03

This file is the decision record for manuscript workstream B. It is independent
of lab deliverable A: evidence produced for A may be reused, but A's operational
acceptance criteria do not define the paper's scientific claim.

## Decision

The current manuscript is **not submission-ready** and should not be polished in
its present form. Its central negative claim (that vision alone is insufficient),
its DINO-depth comparison, and its claim of arbitrary-object generalisation are
not supported strongly enough after the updated literature review and corrected
evaluation.

The defensible paper is narrower and stronger:

> **Measured robot interaction signals disambiguate the visual roles of the
> grasped object and the contact-receiving object among class-agnostic mask
> proposals in open-world, eye-in-hand manipulation.**

The proposed method is a role-conditioned selector over class-agnostic visual
proposals. Gripper state and attachment evidence ground the grasped role. A
measured wrist wrench, camera geometry, depth, and contact timing ground the
contact-receiver role. SAM 2 supplies proposals and tracking; it is not the
scientific contribution.

The paper must not claim that vision-only methods cannot solve the problem. It
may show, through fair baselines and ablations, where measured interaction
signals improve accuracy, selectivity, or data efficiency.

## Current evidence

What is already useful:

- real event detection, class-agnostic proposals, bidirectional propagation, and
  role-tagged output are implemented;
- the correct contact object is present in the proposal pool in 7/7 evaluated
  events, isolating selection rather than segmentation as the current problem;
- the corrected working transform agrees with 6/7 recorded contact events and
  provides a concrete geometric hypothesis to validate independently;
- failed selectors and cue studies are legitimate ablations and diagnostic
  evidence if their protocol remains frozen and their failures are reported.

What is not yet a paper result:

- grasped selection is 0/5 accepted and contact selection is 2/7 accepted under
  A's current selective-prediction gate;
- the recordings are too few and too correlated to support an open-world
  generalisation claim;
- the real camera-to-robot transform has not passed independent calibration;
- no learned or geometric hybrid has yet beaten both vision-only and
  geometry-only baselines on object-held-out data;
- Isaac Sim evidence does not yet exist.

## Position relative to the closest work

| Work | What it already establishes | Remaining distinction B must demonstrate |
|---|---|---|
| Im2Contact (CoRL 2023) | Proprioception plus calibrated RGB-D can predict dense visual contact; the projected end-effector crop is highly informative. | Measured wrist wrench for open-world *instance-role selection* in an eye-in-hand view, without requiring a known grasped-object reference image. |
| DistinctNet / “What's This?” (ICRA 2021) | Robot motion can self-supervise unknown grasped-object segmentation. | Jointly resolve grasped and external contact-receiver roles; show that force geometry adds information beyond motion. |
| MOVES (CVPR 2023) and HOIST-Former (CVPR 2024) | Interaction/hand association can discover and track manipulated objects in human video. | Robot-specific proprioceptive grounding and measured contact, evaluated against adapted interaction-only baselines. |
| HOI-DETR (2026 preprint) | The hand → held object → acted-upon object relation can be detected visually in human video. | The same role structure for a robot under human-to-robot domain shift, with wrench evidence and calibrated geometry. |
| Force-Based Simultaneous Mapping and Object Reconstruction (RA-L 2022) | Wrist force/torque can constrain possible contact locations and update 3-D occupancy during manipulation. | Image-space object identity and temporally propagated role masks rather than occupancy reconstruction. |
| Hoi! (CVPR 2026) | A large synchronized, calibrated force-grounded manipulation dataset exists on essentially the same Bota SensONE + ZED Mini sensing stack. | A role-grounding method and controlled real evaluation; Hoi! is also a candidate external dataset, subject to compatible annotations/views. |

Consequently, the manuscript must remove “to our knowledge” statements about
using wrench geometry for contact localisation unless a narrower claim survives
a complete citation check.

Primary sources:

- Im2Contact: <https://proceedings.mlr.press/v229/kim23b.html>
- DistinctNet: <https://github.com/DLR-RM/DistinctNet>
- MOVES: <https://relh.github.io/moves/>
- HOI-DETR: <https://arxiv.org/abs/2606.17384>
- Force-based mapping: <https://ieeexplore.ieee.org/document/9718152/>
- Hoi!: <https://openaccess.thecvf.com/content/CVPR2026/html/Engelbracht_Hoi_-_A_Multimodal_Dataset_for_Force-Grounded_Cross-View_Articulated_Manipulation_CVPR_2026_paper.html>

## Method to test

Use the same proposal pool for every ablation. For each event, score each mask
with three deliberately separable evidence families:

1. **Vision/interaction:** mask appearance, depth, temporal persistence,
   gripper overlap, attachment motion, and scene-relative motion.
2. **Geometry:** distance and intersection between the mask's 3-D support and
   the measured wrench line of action, with uncertainty induced by calibration,
   synchronization, and sensor noise.
3. **Hybrid:** a calibrated ranker or energy model combining the two, with an
   explicit abstention score.

The primary scientific comparison is hybrid versus vision/interaction-only and
geometry-only. A larger neural model is justified only if this comparison shows
that nonlinear fusion is necessary. Start with an interpretable probabilistic
or shallow learned ranker.

## Isaac Sim's valid role

Isaac Sim is an experimental instrument, not a replacement for the real rig.
It can provide:

- a Franka digital twin with an eye-in-hand RGB-D camera and a virtual wrist
  force/torque measurement;
- exact instance masks, depth, object identities, contact pairs, contact points,
  normals, forces, transforms, and role labels;
- randomized objects, distractors, textures, lighting, camera poses, contact
  locations, friction, mass, stiffness, and approach trajectories;
- controlled corruption of wrench noise/bias, transform error, time offset,
  occlusion, depth dropout, and multiple simultaneous contacts;
- synthetic training data and held-out stress tests that would be impractical to
  label at scale on the physical rig.

It cannot establish real-world robustness, actual Bota/ZED noise, compliance,
latency, or transfer to unseen physical objects. Every principal performance
claim therefore requires a held-out real test set.

The vendored `Data/fr3.urdf` is not standalone: its meshes use
`package://franka_description/...` paths. On the Ubuntu Isaac machine, either
resolve that ROS package during URDF import or use Isaac Sim's maintained Franka
asset, then add the lab's custom gripper bracket, sensor frame, and camera. Do
not assume the stock Franka flange is the `current_pose` subject frame.

Current Isaac Sim supports instance/semantic segmentation and depth annotations,
physics contact/effort sensors, direct simulator ground truth, and domain
randomization. The implementation must pin the installed Isaac Sim version;
sensor and domain-randomization APIs changed in 6.0.

Official references:

- architecture and synthetic annotations: <https://docs.isaacsim.omniverse.nvidia.com/latest/introduction/reference_architecture.html>
- physics sensors: <https://docs.isaacsim.omniverse.nvidia.com/latest/sensors/isaacsim_sensors_physics.html>
- sensor schema: <https://docs.isaacsim.omniverse.nvidia.com/latest/omniverse_usd/sensor_schema.html>

## Experimental design

### Simulation phase

1. Build and hand-check ten deterministic scenes. Verify camera frames,
   projected masks, contact identity, and the sign/units of the reconstructed
   six-axis wrench against analytically simple contacts.
2. Generate scripted grasp–carry–contact episodes with exact synchronized RGB,
   depth, pose, gripper, wrench, instance mask, and role labels. Store the
   generation seed and all randomization parameters.
3. Split by object identity and scene assets, never by adjacent frames. Keep a
   separate stress-test split for calibration and synchronization corruption.
4. Train only on the synthetic training split. Freeze model, thresholds, and
   abstention policy before simulated test and all real evaluation.

The first simulation gate is intentionally bounded: on object-and-scene-held-out
episodes, the hybrid must improve top-1 role selection or coverage at 0.90
precision by at least 10 absolute percentage points over **both** component
baselines. Report confidence intervals and at least three generation/training
seeds. If this gate fails, stop the learned-fusion route; use simulation only to
characterize geometric sensitivity.

### Real phase

Use Monday's A bakeoff only as baseline evidence. The manuscript needs a new,
frozen real protocol after calibration:

- first run a 10–12 demonstration pilot to estimate variance and failure modes;
- target at least 30 independently recorded demonstrations for the main study,
  spanning at least 10 held/receiver object pairings and five scene layouts;
- reserve complete object identities and recording sessions for test;
- include ordinary, visually ambiguous, partially occluded, and failure cases;
- report all exclusions before looking at method scores.

The paper-level success gate is at least 0.90 precision and 0.75 coverage for
both roles on the frozen real held-out split, plus a statistically supported
gain over the strongest fair baseline. Passing A's gate alone is necessary
engineering evidence, not proof of the manuscript hypothesis.

### Baselines and metrics

Required internal baselines:

- existing fixed/region heuristic;
- DINO-depth baseline, retained as one weak baseline rather than the main foil;
- vision/interaction-only proposal ranker;
- wrench-geometry-only selector;
- full hybrid with ablations for wrench, depth, calibration uncertainty, and
  temporal evidence.

External methods should be included only where their inputs can be adapted
fairly. DistinctNet and HOI-DETR are domain-transfer baselines, not drop-in
solutions. An Im2Contact-style end-effector/visual contact baseline is more
important than either for the contact role.

Report role-wise top-1 and top-k accuracy, precision–coverage/selective-risk
curves, mask IoU, track IoU over time, contact-point/ray-to-object distance,
calibration and time-offset sensitivity, and object/scene-held-out results.

## Execution order and stop points

1. Preserve `Docs/twente-paper.tex` as the historical draft. The canonical
   prospective manuscript is `Docs/wrench2role-paper.tex`; its numerical
   results are explicitly marked as projected until replaced by frozen evidence.
2. Convert the literature table above into a complete related-work bibliography
   and remove the invalid negative/novelty claims.
3. On the Ubuntu Isaac machine, record the Isaac Sim version and prove a minimal
   scene can export synchronized RGB, depth, instance masks, transforms, and a
   validated wrench.
4. Run the bounded synthetic geometry-only/vision-only/hybrid test.
5. Continue to a full learned paper only if the simulation gate passes.
6. Calibrate the physical rig, freeze the real protocol, collect independent
   recordings, and perform the final real evaluation.
7. Rewrite the abstract, introduction, and contributions only after steps 4–6
   determine what the evidence actually supports.

## Publication verdict

If the hybrid passes both the simulated held-out gate and the calibrated real
held-out gate, this is a credible RA-L/ICRA-level systems-and-method result:
open-world role grounding from measured interaction, with controlled simulation
analysis and real validation. If only the existing pipeline and 6/7 transform
sanity check remain, it is a useful lab deliverable and possibly a workshop or
demo paper, but not the strong manuscript currently implied by the draft.
