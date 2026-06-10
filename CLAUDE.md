# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Remote collaboration with University of Twente / NAKAMA Robotics Lab. The primary deliverable is a **vision system that identifies task-relevant objects in robot-demonstration data, integrated into the lab's ROS 2 LfD pipeline**. The lab records bags on a Franka FR3 with a Bota wrist F/T sensor, a Franka gripper, and a ZED RGB camera. The lab owns data collection and export; this repo owns the downstream vision module.

Data flow: the lab runs `ros2_unbag` + an in-house merge script on each bag, producing a single pose-synchronised CSV (`<trial>_0.csv`) and per-topic subfolders alongside a ZED PNG folder. We consume that output — we do not maintain the export side anymore. The schema is fixed (`lfdws_t001_0.csv` is the spec).

## Two parallel codebases in this repo

1. **Legacy bag-extraction tooling** — `bag_to_csv.py` (the lab's original) and `unbag_pipeline.py` (a Python replacement, since superseded by `ros2_unbag` on the lab's side). Kept for reference; not the deliverable.
2. **Current vision pipeline** — the `analyze_demo.py` → `segment_events.py` → `propagate_*.py` → `build_sidecar.py` chain, working from the exported CSV+PNGs.

When asked about "the pipeline," the second meaning is current.

## Repo layout

```
Code/        Python scripts (pipeline + figure generators)
Docs/        writeup.tex / .pdf, FAILURE_MODES.md, setup_info.md, reference PDFs
Data/        trial folders + legacy CSV (mostly gitignored)
figures/     generated artifacts referenced by writeup.tex (kept at root)
```

Run everything from the repo root — scripts and `writeup.tex` use cwd-relative
paths to `figures/...` and `Data/...`. Moving `figures/` would break both.

## Three Python venvs, one repo

Different stages need different Python versions / package sets. Each venv is gitignored and dedicated:

| Venv | Python | Used by | Notes |
|---|---|---|---|
| `.venv_analysis` | 3.9 (miniforge) | `Code/analyze_demo.py`, `Code/mask_area_plot.py`, `Code/force_overlay.py`, `Code/build_sidecar.py` | Pandas/numpy/matplotlib only. The miniforge base had a numpy/pandas ABI mismatch; this venv was created clean to work around it. |
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
.venv_analysis/bin/python Code/build_sidecar.py

# optional figures
.venv_analysis/bin/python Code/mask_area_plot.py
.venv_analysis/bin/python Code/force_overlay.py
.venv_sam2/bin/python Code/make_propagation_figure.py
.venv_dado/bin/python Code/_dado_inference.py    # only after Code/run_dado.py set up .venv_dado
```

`Code/identify_objects.py` was the intended single-script end-to-end, but it OOMs on M3 Pro (18 GB unified memory) when both SAM 2 objects share one model state. The working pattern is the separate `Code/propagate_demo.py` + `Code/propagate_cup.py` + `Code/build_sidecar.py` chain above. The JSON contract is identical either way.

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

## Hard-coded knobs you will hit

Three places where the current code is tuned to `lfdws_t001` specifically and will need attention when the second bag arrives:

- `ROLE_SEEDS` in `Code/identify_objects.py` and the `SEED_POINT_FRAC*` constants in `Code/propagate_demo.py` / `Code/propagate_cup.py` — image-relative seed points (e.g. `(0.70, 0.30)` for the grasped object). Placeholders for end-effector projection (`Code/project_ee.py`), which is fully wired but runs DRY until `calibration.yaml` is filled with the lab's ZED intrinsics + `base→camera` extrinsics. The wrench-line variant additionally needs the `bota_frame→base` mount transform (lab hardware, not in any URDF). The Franka Research 3 arm URDF is vendored at `Data/fr3.urdf` (arm-only, flange `fr3_link8`); the Franka Hand TCP is `+0.1034 m` z past the flange, and `current_pose` is published in the `base` frame (the TCP under default FR3+Hand config).
- `GRASP_IMG_ID` / `PRESS_IMG_ID` constants in the propagation scripts — image-timestamp seeds. `Code/identify_objects.py` derives these from the CSV via `detect_events`, so future bags work without changing constants there.
- Event-detection thresholds in `Code/analyze_demo.py` (`detect_events`) — midpoint between open/closed gripper width, and force-peak restricted to the gripper-closed window. Will fail on demos with multiple pick-and-place cycles or no force contact.

See `Docs/FAILURE_MODES.md` for the full honest inventory.

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
- `project_ee.py` — calibrated end-effector + wrench-line projection into the camera image. Reads `calibration.yaml` (camera K + distortion, `base→camera` extrinsic, `bota_frame→base` F/T mount). Runs in DRY mode (reports base-frame geometry, draws nothing) until the calibration blocks are marked `filled: true`, so it never ships fake pixels. EE projection unblocks writeup step 2 (a geometric SAM seed, replacing `auto_seed.py`'s vision-only heuristic); the wrench-line projection (Bicchi 1990: `r0 = (f×τ)/|f|²`, direction `f/|f|`) is the go/no-go pre-test for the wrench-prompted-segmentation research direction.
- `prepare_sam2_frames.py` — one-off util that converts PNGs to `00000.jpg`-style names because SAM 2's `init_state` requires that.
- `download_sam2_ckpt.py` — convenience downloader for the SAM 2 checkpoint.
- `unbag_pipeline.py` / `bag_to_csv.py` — legacy extraction tooling, kept for reference only.

## Conventions from collaborator feedback (apply when extending)

- **No deletions**: don't `rm` scratch / diagnostic / temp files. Leave them in place; let Anurag clean up.
- **Plan before installs**: surface options + tradeoffs (size, model variant) before any `pip install` of large packages or model downloads.
- **Write all scripts first, then run**: don't interleave write→run→write. Use available cores (`torch.set_num_threads(10)` or similar). Logs should be sequential and live (per-frame, per-iteration), not end-of-run summaries.
- **Research bar**: any paper framing of this work must be genuinely novel, not "we composed N existing tools." Systems-integration framings have been explicitly rejected. The deliverable for the lab is the primary work; the paper is additive on top.

- **No personal paths in docs**: never write any user's home directory or absolute filesystem path into a README or any tracked file. All examples must be relative (`<trial>`, `./figures/`, etc.).
