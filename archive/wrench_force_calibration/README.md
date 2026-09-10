# Archived: wrench/force-based camera calibration and geometric contact grounding

**Status: archived, 2026-09-10. Not part of the active deliverable. Kept as
future work — not deleted, not currently maintained.**

## Why this is archived

The lab deliverable is scoped as: **vision-based recognition of objects
relevant to robot tasks, extracted from task demonstration data.** That scope
does not require camera-to-force-sensor calibration — it was pursued because
it was also a promising direction for the (separately archived, see
`MANUSCRIPT_B.md`) research paper, and it consumed roughly six weeks without
producing anything the active pipeline depends on.

**The active production pipeline (`Code/select_objects.py`,
`Code/evaluate_selection.py`, `Code/run_deliverable.py`,
`Code/deliverable_events.py`, `Code/contact_ceiling_study.py`,
`Code/grasp_attachment_study.py`) has zero dependency on anything in this
directory or on `calibration.yaml`'s `bota_to_camera`/`end_effector` blocks.**
This was verified directly before archiving (grepped every import in the
production modules) and confirmed by the full test suite passing unchanged
after the move.

## What's here

Every script, figure, and note from the hand-eye calibration / wrench-ray /
CAD-extraction / current_pose-frame investigation:

- `calibrate_hand_eye.py`, `wrench_ray.py`, `wrench_ray_validate.py`,
  `handeye_offset_search.py`, `verify_camera_intrinsics.py`,
  `verify_board_scale.py`, `generate_charuco_board.py`,
  `extrinsic_grid_search.py` — the ChArUco hand-eye calibration attempt and
  its six independent rejected hypotheses for why it scored 3/7 against the
  real contact events (frame-origin offset, motion blur, ChArUco pattern
  bug, `CharucoBoard` constructor argument order, hand-eye solver method,
  camera distortion). See `LAB_DELIVERABLE_A.md`'s git history for the full
  writeup — that narrative stays in `LAB_DELIVERABLE_A.md` as a record of
  what was tried, even though the acceptance criterion itself is archived.
- `cad_extract_transform.py`, `cad_find_lens_occ.py`, `cad_find_lens_planar.py`,
  `cad_isolate_body_solids.py`, `cad_candidate_sensitivity.py` — the
  earlier, conclusively-exhausted attempt to read `bota_to_camera` off the
  lab's CAD assembly STEP file instead of measuring it.
- `mark_sheet_decode.py`, `mark_sheet_azimuth.py`, `verify_mark_transform.py`,
  `audit_extrinsic.py`, `why_extrinsic.py` — decoding and scoring Mark
  Vlutters' CAD dimension sheet as a `bota_to_camera` candidate.
- `derive_tool_axis.py`, `measure_grasp_offset.py`, `grasp_offset_search.py`,
  `pose_frame_probe.py` — attempts to recover a fixed geometric offset for
  seeding roles without full calibration.
- `contact_seed_pixel.py`, `contact_eval_set.py`, `project_ee.py`,
  `geometric_seed.py`, `grasped_weight_ray.py`,
  `calibration_sensitivity_sweep.py` — the geometric/wrench-projected
  contact-seeding candidate method and its shared evaluation set.
- `sim_wrench_ray_validation.py` — the pre-flight Monte-Carlo sanity check on
  the Bicchi wrench-line recovery math (never a calibration substitute, see
  its own docstring).
- `Docs/` — `calibration_history.md`, the archived draft update to Mark
  about the rotation-convention discrepancy, the sent ChArUco board PDF/PNG,
  and Mark's Franka Desk End-Effector screenshot.
- `figures/` — every plot/CSV this cluster produced.
- Root of this directory also holds the reference CAD/dimension images
  (`bota_sensone_dimensions.png`, `trans_mat_fs_to_cam*.jpg`,
  `rev2_coordinates.png`, `gripper_dimensions.png`) and every
  `calibration_handeye_result*.yaml` + debug overlay this work produced.

## What was NOT moved, and why

`calibration.yaml` stays at the repo root. `Code/seed_scoreboard.py` (the
active cross-seeder evaluation table) still reads it, and the file is the
shared calibration *state* record, not investigation code. Its
`bota_to_camera.filled` stays `false` and its `end_effector` block's
resolved `current_pose_is: bota_origin` fact (confirmed by Mark, real and
worth keeping) stays recorded there.

## If this work is ever resumed

Every script here still runs — imports were verified after the move (each
script's own directory + a path back to the real `Code/event_utils.py` where
needed). Run them the same way, from the repo root:

```bash
.venv_analysis/bin/python archive/wrench_force_calibration/<script>.py ...
```

The three CAD-mesh scripts (`cad_find_lens_occ.py`, `cad_find_lens_planar.py`,
`cad_isolate_body_solids.py`) need `pythonocc-core` (`OCC`), which was never
installed in `.venv_analysis` — that's a pre-existing gap, not something the
move broke.
