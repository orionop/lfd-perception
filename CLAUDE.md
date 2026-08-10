# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Remote collaboration with University of Twente / NAKAMA Robotics Lab. The primary deliverable is a **vision system that identifies task-relevant objects in robot-demonstration data, integrated into the lab's ROS 2 LfD pipeline**. The lab records bags on a Franka FR3 with a Bota wrist F/T sensor, a Franka gripper, and a ZED RGB camera. The lab owns data collection and export; this repo owns the downstream vision module.

Data flow: the lab runs `ros2_unbag` + an in-house merge script on each bag, producing a single pose-synchronised CSV (`<trial>_0.csv`) and per-topic subfolders alongside a ZED PNG folder. We consume that output — we do not maintain the export side anymore. The schema is fixed (`lfdws_t001_0.csv` is the spec).

Exception: `ros2_unbag` currently cannot export bags containing the ZED
`compressedDepth` topic (lab is writing a plugin fix). `Code/mcap_extract.py`
is our own fallback for that case — decodes a raw `.mcap` straight to the
same merged-CSV + PNG-folder layout, no ROS2 install needed. Use it only when
the lab's export is unavailable; once their plugin lands, go back to
consuming their output.

## Two parallel codebases in this repo

1. **Legacy bag-extraction tooling** — `bag_to_csv.py` (the lab's original) and `unbag_pipeline.py` (a Python replacement, since superseded by `ros2_unbag` on the lab's side). Kept for reference; not the deliverable.
2. **Current vision pipeline** — the `analyze_demo.py` → `segment_events.py` → `propagate_*.py` → `build_sidecar.py` chain, working from the exported CSV+PNGs.

When asked about "the pipeline," the second meaning is current.

## Repo layout

```
Code/        Python scripts (pipeline + figure generators)
Docs/        writeup.tex / .pdf, setup_info.md, reference PDFs
Data/        trial folders + legacy CSV (mostly gitignored)
figures/     generated artifacts referenced by writeup.tex (kept at root)
```

Run everything from the repo root — scripts and `writeup.tex` use cwd-relative
paths to `figures/...` and `Data/...`. Moving `figures/` would break both.

## Three Python venvs, one repo

Different stages need different Python versions / package sets. Each venv is gitignored and dedicated:

| Venv | Python | Used by | Notes |
|---|---|---|---|
| `.venv_analysis` | 3.9 (miniforge) | `Code/analyze_demo.py`, `Code/mask_area_plot.py`, `Code/force_overlay.py`, `Code/build_sidecar.py`, `Code/mcap_extract.py` | Pandas/numpy/matplotlib only, plus `mcap`/`mcap-ros2-support` (added for `mcap_extract.py`, pure-Python, no ROS2 needed). The miniforge base had a numpy/pandas ABI mismatch; this venv was created clean to work around it. |
| `.venv_sam2` | 3.11 | `Code/propagate_demo.py`, `Code/propagate_cup.py`, `Code/identify_objects.py`, `Code/prepare_sam2_frames.py` | SAM 2 requires Python ≥3.10. Includes torch 2.12 with MPS, plus `sam2` from GitHub, plus pandas (added late — was the cause of one crash). |
| `.venv_dado` | 3.11 | `Code/_dado_inference.py` (invoked via `Code/run_dado.py`) | DINOv2 + Depth-Anything-V2 via `transformers`. The DADO orchestrator (`Code/run_dado.py`) creates this venv if missing. |

Do not unify them. Mixing SAM 2 deps into the analysis venv breaks pandas; mixing pandas into the DADO venv is fine but pointless.

## End-to-end run on a new bag

Assuming a bag has been exported into `<trial_dir>/` with the standard layout:

```bash
# 1. event detection + timeline figure (Result A) + raw event-frame strip (Result B)
.venv_analysis/bin/python Code/analyze_demo.py --trial <trial_dir>

# 2. convert PNG frames to zero-padded JPGs (SAM 2 video predictor requirement)
.venv_sam2/bin/python Code/prepare_sam2_frames.py \
    --src <trial_dir>/zed_zed_node_rgb_color_rect_image_compressed \
    --dst frames_jpg

# 3. propagate the grasped object (carrot) forward from the grasp event
.venv_sam2/bin/python Code/propagate_demo.py \
    --trial <trial_dir> --ckpt sam2.1_hiera_large.pt --jpg_dir frames_jpg

# 4. propagate the contact-receiving object (cup) bidirectionally from the press
.venv_sam2/bin/python Code/propagate_cup.py \
    --trial <trial_dir> --ckpt sam2.1_hiera_large.pt --jpg_dir frames_jpg

# 5. compose the JSON sidecar + summary CSV + merged overlay MP4 from steps 3+4
.venv_analysis/bin/python Code/build_sidecar.py \
    --trial <trial_dir> \
    --carrot_csv figures/propagation_summary.csv \
    --cup_csv figures/propagation_cup_summary.csv \
    --out figures/identify

# optional figures
.venv_analysis/bin/python Code/mask_area_plot.py --trial <trial_dir>
.venv_analysis/bin/python Code/force_overlay.py --trial <trial_dir>
.venv_sam2/bin/python Code/make_propagation_figure.py
.venv_dado/bin/python Code/_dado_inference.py    # only after Code/run_dado.py set up .venv_dado
```

`Code/identify_objects.py` was the intended single-script end-to-end, but it OOMs on M3 Pro (18 GB unified memory) when both SAM 2 objects share one model state. The working pattern is the separate `Code/propagate_demo.py` + `Code/propagate_cup.py` + `Code/build_sidecar.py` chain above. The JSON contract is identical either way.

Both propagation scripts take `--offload_video_to_cpu` (off by default). `lfdws_t001`'s 497 frames fit in device memory without it; `lfdws_t001_depth`'s 1013 frames do not (`init_state` tries to allocate the whole decoded video in one buffer — hit `RuntimeError: Invalid buffer size: 11.87 GiB` on MPS without the flag). Pass it for any trial with enough frames to exceed available device memory.

`propagate_demo.py`, `propagate_cup.py`, `propagate_demo_bidir.py`, `analyze_demo.py`, and `build_sidecar.py` derive their merged MP4/CSV/JSON output paths from `--out` and back up any pre-existing file at that path to `<path>.bak` before overwriting, so different trials' outputs never silently clobber each other. Use a distinct `--out` per trial (e.g. `figures/propagation_cup_depth`, `figures/identify_depth`) so multiple trials' artifacts coexist.

`build_sidecar.py`, `mask_area_plot.py`, and `force_overlay.py` all take `--trial` and detect events via the same force-only fallback as `auto_seed.py` (see below) — they work on trials without a gripper topic, just with grasp/release omitted. `force_overlay.py` additionally needs a real `--carrot_csv` (grasped-object trajectory) to fit its base→uv regression; on a trial with no grasp event it exits cleanly with `[fatal] too few pairs to fit projection` rather than producing bad output.

The symmetric gap — a gripper topic but no wrench topic at all (confirmed on `lfdws_t004`/`lfdws_t005`: the F/T sensor wasn't publishing) — is also handled, in `analyze_demo.py`, `auto_seed.py`, `build_sidecar.py`, `mask_area_plot.py`, `multi_event.py`, and `force_overlay.py`: each checks for the wrench columns and falls back to grasp/release-only detection (no `press` event, `contact_receiver` role skipped) when they're absent. `force_overlay.py` exits cleanly (`[fatal] no wrench topic...`) rather than crashing, same pattern as its carrot-missing case. Every one of these fallbacks was regression-tested against `lfdws_t001` (both sensors) and `lfdws_t001_depth` (force, no gripper) before trusting it — zero behaviour change confirmed on both.

`analyze_demo.py`'s default `--out` is the fixed `figures/` dir (intentional — `lfdws_t001`'s canonical `figures/timeline.png`/`event_frames.png` are referenced throughout the writeup and must keep that exact path). Running it on a second trial without an explicit `--out` will silently target the same files, so it now backs up any pre-existing file at the output path to `<path>.bak` first, same as the rest of the pipeline — but pass a trial-specific `--out` for any trial other than the primary one.

## Model checkpoints (not in git)

- `sam_vit_h_4b8939.pth` (2.4 GB) — SAM 1 ViT-H. Download from `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`. Used by `segment_events.py`.
- `sam2.1_hiera_large.pt` (~900 MB) — SAM 2.1 Hiera-L. `download_sam2_ckpt.py` fetches it from `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt`.

Both must sit in the repo root for the scripts' default `--ckpt` paths.

## Trial data layout (what scripts assume)

Trial directories live under `Data/` and are structured exactly as the lab's `ros2_unbag` export:

```
Data/<trial>/
├── <trial>_0.csv                                            # merged, dot-separated, topic-prefixed
├── metadata.yaml
├── config_unbag_to_*.json                                   # ros2_unbag configs
├── bota_post_wrench_body_compensated/*.csv                  # per-topic
├── NS_1_franka_gripper_joint_states/*.csv
├── NS_1_franka_robot_state_broadcaster_current_pose/*.csv
└── zed_zed_node_rgb_color_rect_image_compressed/
    └── {nanosecond_timestamp}.png                           # 497 frames in lfdws_t001
```

The merged CSV's image column is the literal PNG filename (without `.png`). Per-topic subfolders are redundant with the merged CSV; scripts read only the merged CSV + PNG folder.

`Data/lfdws_t001/` is the only image-bearing trial currently present (gitignored, ~530 MB). `Data/lfdws_trial_002/` is an older 2-topic trial without images.

`Data/lfdws_t001_depth/` (gitignored) was produced by `mcap_extract.py` from the raw
bag Mark sent when `ros2_unbag` failed on the depth topic. Same layout, plus a
`zed_zed_node_depth_depth_registered_compressedDepth/` folder of 16-bit millimetre
PNGs (0 = invalid pixel). This trial's `metadata.yaml` has no `franka_gripper/
joint_states` topic — confirm with the lab whether that's expected before running
`analyze_demo.py`'s gripper-width event detection on it.

`Data/lfdws_t004/` and `Data/lfdws_t005/` (gitignored) are two more `mcap_extract.py`
extractions, the opposite gap from `lfdws_t001_depth`: both have a gripper topic
(real grasp/release detection works) but `bota_post/wrench_body_compensated`
published 0 messages and `bota_driver_node/wrench` isn't in the bag at all — no
force data whatsoever, so `press` events are never detected (gripper-only
fallback, see below). Both have RGB + depth. `lfdws_t004` is 1745 frames / 116.8s;
`lfdws_t005` is 726 frames / 49.0s. Neither is the carrot/cup or plate/screwdriver
scene — looks like a hinge/latch-and-fastener manipulation task on a wall pegboard.

## Hard-coded knobs you will hit

Three places where the current code is tuned to `lfdws_t001` specifically and will need attention when the second bag arrives:

- `ROLE_SEEDS` in `Code/identify_objects.py` and the `SEED_POINT_FRAC*` constants in `Code/propagate_demo.py` / `Code/propagate_cup.py` — image-relative seed points (e.g. `(0.70, 0.30)` for the grasped object). Placeholders for end-effector projection (`Code/project_ee.py`), which is fully wired but runs DRY until `calibration.yaml` is filled with the lab's ZED intrinsics + `base→camera` extrinsics. The wrench-line variant additionally needs the `bota_frame→base` mount transform (lab hardware, not in any URDF). The Franka Research 3 arm URDF is vendored at `Data/fr3.urdf` (arm-only, flange `fr3_link8`); the Franka Hand TCP is `+0.1034 m` z past the flange, and `current_pose` is published in the `base` frame (the TCP under default FR3+Hand config).
- `GRASP_IMG_ID` / `PRESS_IMG_ID` constants in the propagation scripts — image-timestamp seeds. `Code/identify_objects.py` derives these from the CSV via `detect_events`, so future bags work without changing constants there.
- Event-detection thresholds in `Code/analyze_demo.py` (`detect_events`) — midpoint between open/closed gripper width, and force-peak restricted to the gripper-closed window. Will fail on demos with multiple pick-and-place cycles or no force contact.
- `Code/auto_seed.py`'s `score_mask` role priors (area fraction cap `< 0.4`, lower-half-of-frame bonus for `contact_receiver`) are tuned to `lfdws_t001`'s object scale and will reject a correct contact-receiver that's larger than 40% of the frame (e.g. a plate rather than a cup). Reseed manually when that happens; the durable fix is `project_ee.py`'s geometric EE-projection seed once calibration lands.

## What the writeup is

`Docs/writeup.tex` / `Docs/writeup.pdf` is the running progress doc shared with the lab. The original 4 pages are the first version; later sections append updates. **Do not edit the original 4 pages** — only append after the existing content. Compile from the repo root:

```bash
pdflatex -output-directory=Docs Docs/writeup.tex
```

Run twice to resolve refs. `.aux` / `.log` / `.out` are byproducts that can be left alone; only `writeup.tex` and `writeup.pdf` matter.

## What scripts are doing at a high level

(All under `Code/`.)

- `analyze_demo.py` — event detection from the merged CSV; produces `figures/timeline.png` and `figures/event_frames.png`.
- `segment_events.py` — SAM 1 ViT-H on the three event frames with configurable point prompts.
- `propagate_demo.py` / `propagate_cup.py` — SAM 2 video predictor seeded at one proprioceptive event, writes per-frame overlays + summary CSV + MP4 for one object.
- `build_sidecar.py` — pure post-processing: reads both propagation CSVs and produces the the lab-facing JSON (`figures/identify/objects.json`) + per-frame combined overlays + MP4.
- `identify_objects.py` — intended one-shot end-to-end (currently OOMs on this hardware; see above).
- `combined_strip.py` / `make_propagation_figure.py` / `mask_area_plot.py` / `force_overlay.py` — figure generators for the writeup.
- `run_dado.py` + `_dado_inference.py` — DADO-style label-free baseline (DINOv2 attention × Depth-Anything depth). Negative-result figure for the writeup.
- `project_ee.py` — calibrated end-effector + wrench-line projection into the camera image, reworked for the eye-in-hand rig (camera mounted on the gripper, not fixed in the world — see `calibration.yaml`'s header). Per-frame extrinsics: `T_base_camera(t) = T_base_bota(t) @ T_bota_camera`, where `T_base_bota(t)` comes straight from `current_pose` and `T_bota_camera` is the one fixed unknown. Reads `calibration.yaml` (camera K + distortion, `bota_to_camera`). Runs in DRY mode (reports base-frame geometry, draws nothing) until both blocks are marked `filled: true`, so it never ships fake pixels. EE projection unblocks a geometric SAM seed, replacing `auto_seed.py`'s vision-only heuristic; the wrench-line projection (Bicchi 1990: `r0 = (f×τ)/|f|²`, direction `f/|f|`) is the go/no-go pre-test for the wrench-prompted-segmentation research direction.
- `calibrate_hand_eye.py` — recovers `bota_to_camera` by direct hand-eye calibration (OpenCV `cv2.calibrateHandEye`, AX=XB, eye-in-hand) instead of reading it off the CAD drawings. Supersedes `cad_extract_transform.py`/`cad_find_lens_occ.py` — the CAD reading left the lens position ambiguous between two 63mm-apart candidates, and `cad_candidate_sensitivity.py` showed both candidates project the wrench ray entirely outside the image on every trial tested (rotation is unreliable, not just the translation). Needs a physically-measured ChArUco board and a short recording of the arm pausing at several distinct poses in front of it (board fixed in the world, not held by the gripper). Both the ArUco/ChArUco detection stage and the AX=XB solve were validated against synthetic data before ever touching real captures (0.0° / ~1e-16m recovery error on a synthetic ground-truth transform) — the math is known-correct; real rig time only needs to validate detection quality. Prints the result for manual review; never writes `calibration.yaml` automatically.
- `prepare_sam2_frames.py` — one-off util that converts PNGs to `00000.jpg`-style names because SAM 2's `init_state` requires that.
- `download_sam2_ckpt.py` — convenience downloader for the SAM 2 checkpoint.
- `unbag_pipeline.py` / `bag_to_csv.py` — legacy extraction tooling, kept for reference only.
- `mcap_extract.py` — our fallback `.mcap` -> merged-CSV extractor for bags `ros2_unbag` can't export (currently: anything with the ZED `compressedDepth` topic). Pure Python via `mcap-ros2-support`, no ROS2 install. Decodes `WrenchStamped`/`PoseStamped`/`CompressedImage` directly (schemas are embedded in the `.mcap` file) and manually decodes the `compressedDepth` transport (12-byte header of 3 float32s — format flag, depthQuantA, depthQuantB — then a PNG; recovers `depth_m = depthQuantA / (code - depthQuantB)`, written out as 16-bit millimetre PNGs). Merges all topics onto the `current_pose` timeline via `pandas.merge_asof(..., direction="nearest")`. Output layout matches the documented trial schema so the rest of the pipeline runs unmodified. `flatten()` writes a scalar list field (e.g. `JointState.position`) as ONE bracketed-string column (`"[0.03, 0.03]"`), matching `ros2_unbag`'s convention that `ast.literal_eval`-based parsing downstream expects — it originally exploded these into indexed `name[0]`/`name[1]` columns instead, which silently broke gripper-width parsing the first time a gripper-bearing bag actually went through this extractor (`lfdws_t004`/`lfdws_t005`); fixed, and every event-detection function across the pipeline was regression-tested against both existing trials before trusting the fix.
- `auto_seed.py` — vision-only SAM-automatic-mask seed picker (see "Hard-coded knobs" below). Falls back to a force-only single-press event when the trial has no gripper topic (skips the `grasped` role entirely — nothing to seed without a grasp event).
- `force_only_events.py` — standalone force-peak detector for gripper-less trials (no held-window restriction, since there's no grasp/release to bound it). Companion to `multi_event.py` for trials without a gripper topic.
- `multi_event.py` — detects ALL grasp/release/press events (not just one of each) and groups them into interaction cycles; needs a gripper topic.
- `object_identity.py` — DINOv2-embedding + agglomerative clustering to give tracked objects stable identities across frames/demos.
- `trial_report.py` — one diagnostic PDF per trial (`Docs/reports/report_<trial>.pdf`), pulling in whichever figures exist for that trial plus the `--sidecar_json` summary table.

## Conventions from collaborator feedback (apply when extending)

- **No deletions**: don't `rm` scratch / diagnostic / temp files. Leave them in place; let Anurag clean up.
- **Plan before installs**: surface options + tradeoffs (size, model variant) before any `pip install` of large packages or model downloads.
- **Write all scripts first, then run**: don't interleave write→run→write. Use available cores (`torch.set_num_threads(10)` or similar). Logs should be sequential and live (per-frame, per-iteration), not end-of-run summaries.
- **Research bar**: any paper framing of this work must be genuinely novel, not "we composed N existing tools." Systems-integration framings have been explicitly rejected. The deliverable for the lab is the primary work; the paper is additive on top.

- **No personal paths in docs**: never write any user's home directory or absolute filesystem path into a README or any tracked file. All examples must be relative (`<trial>`, `./figures/`, etc.).
