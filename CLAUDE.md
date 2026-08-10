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
2. **Current vision pipeline** — `analyze_demo.py` (events) → `prepare_sam2_frames.py` → `auto_seed.py` (seed picking) → `propagate_demo_bidir.py` / `propagate_object_n.py` (SAM 2 propagation, N objects) → `build_sidecar_multi.py` (canonical JSON sidecar). Working from the exported CSV+PNGs. `segment_events.py` (SAM 1) and the older `propagate_demo.py`/`propagate_cup.py`/`build_sidecar.py`/`identify_objects.py` two-role chain are earlier iterations, kept for reference but superseded — see "End-to-end run on a new bag" below.

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
| `.venv_sam2` | 3.11 | `Code/propagate_demo_bidir.py`, `Code/propagate_object_n.py`, `Code/auto_seed.py`, `Code/prepare_sam2_frames.py` (also the older `propagate_demo.py`/`propagate_cup.py`/`identify_objects.py`) | SAM 2 requires Python ≥3.10. Includes torch 2.12 with MPS, plus `sam2` from GitHub, plus pandas (added late — was the cause of one crash). |
| `.venv_dado` | 3.11 | `Code/_dado_inference.py` (invoked via `Code/run_dado.py`) | DINOv2 + Depth-Anything-V2 via `transformers`. The DADO orchestrator (`Code/run_dado.py`) creates this venv if missing. |

Do not unify them. Mixing SAM 2 deps into the analysis venv breaks pandas; mixing pandas into the DADO venv is fine but pointless.

## End-to-end run on a new bag

Assuming a bag has been exported into `<trial_dir>/` with the standard layout:

```bash
# 1. event detection + timeline figure + raw event-frame strip
.venv_analysis/bin/python Code/analyze_demo.py --trial <trial_dir> --out <fig_dir>

# 2. convert PNG frames to zero-padded JPGs (SAM 2 video predictor requirement)
.venv_sam2/bin/python Code/prepare_sam2_frames.py \
    --src <trial_dir>/zed_zed_node_rgb_color_rect_image_compressed \
    --dst frames_jpg

# 3. auto-pick a SAM 2 seed point per role (no hard-coded image fractions)
.venv_sam2/bin/python Code/auto_seed.py --trial <trial_dir> --ckpt sam_vit_h_4b8939.pth \
    --out_csv <fig_dir>/auto_seed.csv --out_overlay <fig_dir>/auto_seed_overlay.png

# 4. propagate grasped + contact_receiver bidirectionally (add --offload_video_to_cpu
#    for trials with enough frames to exceed device memory -- see below)
.venv_sam2/bin/python Code/propagate_demo_bidir.py \
    --trial <trial_dir> --ckpt sam2.1_hiera_large.pt --jpg_dir frames_jpg \
    --out <fig_dir>/propagation_grasped
# additional objects beyond the first two roles: seed manually, use --seed_box
# instead of --seed_x/--seed_y for multi-coloured objects (see "Hard-coded knobs")
.venv_sam2/bin/python Code/propagate_object_n.py \
    --trial <trial_dir> --ckpt sam2.1_hiera_large.pt --jpg_dir frames_jpg \
    --obj_id 3 --role <role_name> --seed_img_id <id> --seed_box "x0,y0,x1,y1" \
    --out <fig_dir>/propagation_obj3

# 5. compose the JSON sidecar + summary CSV + merged overlay MP4 (any object count)
.venv_analysis/bin/python Code/build_sidecar_multi.py \
    --trial <trial_dir> \
    --object "1:grasped:<fig_dir>/propagation_grasped_summary.csv:0,255,0" \
    --object "2:contact_receiver:<fig_dir>/propagation_cup_summary.csv:255,0,255" \
    --out <fig_dir>/identify

# optional figures
.venv_analysis/bin/python Code/mask_area_plot.py --trial <trial_dir>
.venv_analysis/bin/python Code/force_overlay.py --trial <trial_dir>
.venv_sam2/bin/python Code/make_propagation_figure.py
.venv_dado/bin/python Code/_dado_inference.py    # only after Code/run_dado.py set up .venv_dado
.venv_analysis/bin/python Code/trial_report.py --trial <trial_dir> \
    --sidecar_json <fig_dir>/identify/objects.json --fig_dir <fig_dir>   # one diagnostic PDF
```

`Code/build_sidecar_multi.py` is the canonical sidecar builder — it takes any number of `--object obj_id:role:summary_csv:bgr_color` entries, so single-object and 4-object trials go through the same tool. `Code/build_sidecar.py` (fixed two-role: grasped + contact_receiver) and `Code/propagate_demo.py`/`Code/propagate_cup.py` (the two scripts it composes) are the earlier iteration — kept for reference/backward compat, not what to reach for on a new trial. `Code/identify_objects.py` was the intended single-script end-to-end but OOMs on M3 Pro (18 GB unified memory) when multiple SAM 2 objects share one model state; the split propagate-per-object pattern above is the working one. The JSON contract is the same across all of them.

Step 3 (`auto_seed.py`) is optional: if the seed CSV doesn't exist, the propagation scripts fall back to hard-coded defaults tuned to the original trial — expect a manual reseed on a sufficiently different scene (see `Code/auto_seed.py`'s docstring and "Hard-coded knobs" below).

Both propagation scripts take `--offload_video_to_cpu` (off by default). `lfdws_t001`'s 497 frames fit in device memory without it; `lfdws_t001_depth`'s 1013 frames do not (`init_state` tries to allocate the whole decoded video in one buffer — hit `RuntimeError: Invalid buffer size: 11.87 GiB` on MPS without the flag). Pass it for any trial with enough frames to exceed available device memory.

`propagate_demo_bidir.py`, `propagate_object_n.py`, `analyze_demo.py`, `build_sidecar_multi.py` (and the older `propagate_demo.py`/`propagate_cup.py`/`build_sidecar.py`) derive their merged MP4/CSV/JSON output paths from `--out` and back up any pre-existing file at that path to `<path>.bak` before overwriting, so different trials' outputs never silently clobber each other. Use a distinct `--out` per trial (e.g. `figures/propagation_cup_depth`, `figures/identify_depth`) so multiple trials' artifacts coexist.

`build_sidecar_multi.py` (and the older `build_sidecar.py`), `mask_area_plot.py`, and `force_overlay.py` all take `--trial` and detect events via the same force-only fallback as `auto_seed.py` (see below) — they work on trials without a gripper topic, just with grasp/release omitted. `force_overlay.py` additionally needs a real `--carrot_csv` (grasped-object trajectory) to fit its base→uv regression; on a trial with no grasp event it exits cleanly with `[fatal] too few pairs to fit projection` rather than producing bad output.

The symmetric gap — a gripper topic but no wrench topic at all (confirmed on `lfdws_t004`/`lfdws_t005`: the F/T sensor wasn't publishing) — is also handled, in `analyze_demo.py`, `auto_seed.py`, `build_sidecar_multi.py`/`build_sidecar.py`, `mask_area_plot.py`, `multi_event.py`, and `force_overlay.py`: each checks for the wrench columns and falls back to grasp/release-only detection (no `press` event, `contact_receiver` role skipped) when they're absent. `force_overlay.py` exits cleanly (`[fatal] no wrench topic...`) rather than crashing, same pattern as its carrot-missing case. Every one of these fallbacks was regression-tested against `lfdws_t001` (both sensors) and `lfdws_t001_depth` (force, no gripper) before trusting it — zero behaviour change confirmed on both.

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

`Data/lfdws_trial_002/` is an older 2-topic trial without images, not part of the vision pipeline. All other trials below are gitignored (bulk binary data, not in git); each also needs its `figures/<trial>/` output dir added to `.gitignore`'s bulk-artifact section when first processed (missed once for `t001_labexport`/`t002_labexport` — caught it staging thousands of per-frame PNGs, don't repeat that).

| Trial | Sensors | Scene | Notes |
|---|---|---|---|
| `lfdws_t001` | gripper + F/T, no depth | carrot (grasped) / cup (contact_receiver) | canonical trial; `figures/timeline.png` etc. reference this one by fixed path |
| `lfdws_t001_depth` | F/T only, no gripper topic | plate / screwdriver / charger (4 objects) | `mcap_extract.py` extraction; real depth |
| `lfdws_t002_new` | gripper + F/T | Rubik's cube | `mcap_extract.py` extraction, predates the lab's native depth export; same underlying bag as `t002_labexport` |
| `lfdws_t004`, `lfdws_t005` | gripper only, **no F/T at all** | hinge/latch/fastener on a wall pegboard | `mcap_extract.py` extractions; `press` never detected (gripper-only fallback) |
| `lfdws_t001_labexport` | gripper + F/T | latch/hinge contact (short 9.7s clip, no real grasp cycle) | lab's own native `ros2_unbag` export (first with working depth plugin); nested `Data/lfdws_t001_labexport/lfdws_t001/` layout |
| `lfdws_t002_labexport` | gripper + F/T | **3 phases in one recording**: latch contact → cube pick-press-drop → latch contact again | lab's native export; same bag as `lfdws_t002_new`. A naive single-event (global-max-force) detector picks the wrong phase's event here — see "Hard-coded knobs" |

`lfdws_t001_depth` produced by `mcap_extract.py` decodes `compressedDepth` into a `zed_zed_node_depth_depth_registered_compressedDepth/` folder of 16-bit millimetre PNGs (0 = invalid pixel); the lab's native exports (`_labexport` trials) additionally include float32-metre `.npy` depth alongside the PNG.

## Hard-coded knobs / known failure modes you will hit

- **`Code/analyze_demo.py`'s `detect_events()` has no window restriction on the force-peak search** — it reports the single global-maximum force peak across the *whole* recording. This silently attributes the wrong phase's contact to the wrong event on any multi-cycle/multi-phase recording (confirmed on `lfdws_t002_labexport`, whose largest force peak is in an unrelated latch-contact phase, not the cube task). `Code/build_sidecar_multi.py`, `Code/auto_seed.py`, and `Code/multi_event.py` are unaffected — they all restrict the press search to the window between the detected grasp and release events. Don't trust `analyze_demo.py`'s `timeline.png`/`event_frames.png` at face value on a multi-cycle recording; cross-check with `multi_event.py` or by inspecting the seeded frame directly.
- **Point vs. box SAM 2 prompts**: a point prompt on a multi-coloured object (e.g. a Rubik's cube) segments only the locally-contiguous coloured region under it, not the physical object. Worse, propagated across a full recording, a point-seeded track that loses the object (goes out of frame) drifts catastrophically onto unrelated background (measured: mean 46% of frame, peaking at 95%) rather than degrading to empty like a box-seeded track does. Use `--seed_box` (supported by `Code/propagate_object_n.py`) instead of `--seed_x`/`--seed_y` for any multi-coloured or irregularly-textured object.
- `Code/auto_seed.py`'s `score_mask` role priors (area fraction cap `< 0.4`, lower-half-of-frame bonus for `contact_receiver`) are tuned to `lfdws_t001`'s object scale and will reject a correct contact-receiver that's larger than 40% of the frame (e.g. a plate rather than a cup), or pick an empty-background mask on a multi-phase recording whose press event isn't near the true object. Reseed manually when that happens.
- **`calibration.yaml`'s `bota_to_camera` is still unresolved.** Two CAD-derived candidate lens positions (63mm apart, matching the ZED Mini's stereo baseline) were tested directly via `Code/cad_candidate_sensitivity.py`: both project the wrench ray entirely outside the image on every trial tested, meaning the CAD-derived *rotation* is unreliable, not just the ambiguous translation. `Code/calibrate_hand_eye.py` (OpenCV hand-eye calibration, `cv2.calibrateHandEye`) is the intended path forward — needs real rig access with a physically-measured ChArUco board, not yet run on real data. `Code/sim_wrench_ray_validation.py` is a pre-flight sanity check on the recovery *math* only (confirms Bota SensONE's real sensor noise isn't the bottleneck) — it does NOT validate calibration and must never be cited as if it does; the camera extrinsic is defined exactly in that simulation, which sidesteps the actual open problem.
- The Franka Research 3 arm URDF is vendored at `Data/fr3.urdf` (arm-only, flange `fr3_link8`); the Franka Hand TCP is `+0.1034 m` z past the flange. `current_pose` is published in the base frame, but as of the eye-in-hand rig-model correction it reports the **Bota SensONE origin**, not the TCP (see `calibration.yaml`'s header) — `bota_to_tcp` in `calibration.yaml`'s `end_effector` block is still `null`.

## What the writeup is

`Docs/writeup.tex` / `Docs/writeup.pdf` is the running progress doc shared with the lab. The original 4 pages are the first version; later sections append updates. **Do not edit the original 4 pages** — only append after the existing content. Compile from the repo root:

```bash
pdflatex -output-directory=Docs Docs/writeup.tex
```

Run twice to resolve refs. `.aux` / `.log` / `.out` are byproducts that can be left alone; only `writeup.tex` and `writeup.pdf` matter.

`Docs/publication.tex` is a separate document: an arXiv/conference-style manuscript draft (the additive paper direction, not the lab deliverable — see "Research bar" below). Same compile pattern. Every number in it must trace to an actual run; when adding a result, verify it against the source script's output before writing it into the LaTeX, and check for stale cross-references/counts elsewhere in the document afterward (e.g. changing the trial count in one table without updating a prose sentence stating the old count is a real recurring mistake here).

`Docs/reports/report_<trial>.pdf` are per-trial diagnostic PDFs generated by `Code/trial_report.py` (pulls in whichever figures exist for that trial plus the sidecar summary table) — gitignored, regenerable, not hand-edited.

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
- `object_identity.py` — DINOv2-embedding + agglomerative clustering to give tracked objects stable identities within one trial. `object_identity_cross_trial.py` pools crops across multiple trials to test whether identities generalize (repeated finding: they don't cleanly — fragments and cross-contaminates even after `object_identity_cross_trial_sweep.py` swept 6 distance thresholds × 2 crop styles with zero clean configs). This is a documented negative result, not something to keep re-running unmodified expecting a different outcome; not required for the lab deliverable either (SAM 2's per-trial `obj_id` already gives reliable within-trial identity, which is what the pipeline actually needs).
- `trial_report.py` — one diagnostic PDF per trial (`Docs/reports/report_<trial>.pdf`), pulling in whichever figures exist for that trial plus the `--sidecar_json` summary table.
- `build_sidecar_multi.py` — canonical sidecar builder (see "End-to-end run on a new bag"); takes any number of `--object obj_id:role:summary_csv:bgr_color` specs. Its `mask_from_overlay()` checks all 3 BGR channels against the target color, which is stricter (and more correct — excludes anti-aliasing edge noise) than `build_sidecar.py`'s legacy 1-2 channel check.
- `propagate_object_n.py` / `propagate_demo_bidir.py` — generic bidirectional SAM 2 propagation for the Nth object in a trial; `--seed_box` support (vs. point-only) is what fixes the multi-coloured-object seeding failure above.
- `cad_extract_transform.py` / `cad_find_lens_occ.py` — STEP-file assembly-tree parsing and cylindrical-bore mining, the (abandoned) attempt to derive `bota_to_camera` from CAD drawings instead of measurement. Kept for reference; see "Hard-coded knobs" for why this didn't work.
- `cad_candidate_sensitivity.py` — runs both CAD-derived candidate `bota_to_camera` transforms live and reports how far off the projected wrench ray lands vs. ground truth; the result that ruled out the CAD approach.
- `sim_wrench_ray_validation.py` — analytical Monte-Carlo pre-flight check on the Bicchi recovery math only (not a calibration substitute — see "Hard-coded knobs").
- `event_detection_accuracy_table.py` — recomputes event detection live across every trial and tabulates sensor profile / detected events / pass-fail, replacing prose claims about detector robustness with a checkable table.
- `_dado_vs_groundtruth_t002labexport.py` / `_dado_vs_groundtruth_all_trials.py` — scores the DADO-style label-free proposer (DINOv2 attention × real depth) against the propagated ground-truth object mask, per event, across all trials with both real depth and a ground-truth mask. IoU is consistently low (~0.16 mean across 16 events) — the quantitative version of the qualitative DADO negative result.
- `force_only_multi_event.py` — the force-only analogue of `multi_event.py`: groups `force_only_events.py`'s force-peaks into activity clusters by time-gap for trials with no gripper topic (no grasp/release to bound cycles otherwise). Companion, not a superseded duplicate.
- `presence_signal.py` — standalone diagnostic: bbox-diagonal + centroid presence signal, more robust to partial occlusion than raw mask-pixel-count for judging whether a tracked object is genuinely present in a frame. Reads `objects_summary.csv`, doesn't touch the sidecar builders.
- `auto_seed_depth_prior.py` — standalone experiment testing whether real per-pixel depth (depth-bearing trials only) as a near-camera prior improves `auto_seed.py`'s scoring; does not modify `auto_seed.py`, degrades to a no-op on trials without depth.

## Conventions from collaborator feedback (apply when extending)

- **No deletions**: don't `rm` scratch / diagnostic / temp files. Leave them in place; let Anurag clean up.
- **Plan before installs**: surface options + tradeoffs (size, model variant) before any `pip install` of large packages or model downloads.
- **Write all scripts first, then run**: don't interleave write→run→write. Use available cores (`torch.set_num_threads(10)` or similar). Logs should be sequential and live (per-frame, per-iteration), not end-of-run summaries.
- **Research bar**: any paper framing of this work must be genuinely novel, not "we composed N existing tools." Systems-integration framings have been explicitly rejected. The deliverable for the lab is the primary work; the paper is additive on top.
- **Simulation is not a substitute for real-hardware validation** when the paper's claim is specifically about overcoming a real-world measurement/calibration difficulty (see `sim_wrench_ray_validation.py`'s docstring for why). A simulation that defines the exact quantity the paper is trying to measure doesn't validate the claim — it sidesteps it. Fine as a cheap pre-flight sanity check, never as evidence.
- **No personal paths in docs**: never write any user's home directory or absolute filesystem path into a README or any tracked file. All examples must be relative (`<trial>`, `./figures/`, etc.).
- **Git commit messages must not include a `Co-Authored-By: Claude` trailer** in this repo — omit it entirely, just the summary + body.
