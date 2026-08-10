# RLBD — Robot Learning from Demonstration

Vision module for the lab's ROS 2 LfD pipeline. Given a demonstration bag
(robot pose, gripper, force/torque, ZED RGB-D), it identifies task-relevant
objects per interaction phase and writes a JSON sidecar consumable by
downstream LfD code.

## Pipeline

![Pipeline overview](figures/pipeline.png)

Proprioceptive event detection on the merged CSV identifies grasp, release,
and force-contact moments (gracefully degrading when either the gripper or
the F/T sensor is absent from a given bag). Each event indexes the
corresponding ZED frame, which seeds SAM 2 (frozen) with a point or box
prompt. Bidirectional video propagation yields per-frame, role-tagged masks
for an arbitrary number of tracked objects, aggregated into a JSON sidecar.

## Rig

Franka Research 3 arm, Franka Hand gripper, Bota SensONE wrist F/T sensor,
ZED Mini RGB-D camera. The camera is **eye-in-hand**: mounted on a bracket
bolted to the gripper, not fixed in the world. `current_pose` is the pose of
the Bota sensor's own origin in the robot base frame (not the fingertip
TCP), so the camera's pose in the base frame is a per-frame quantity derived
from a single fixed `bota→camera` bracket transform. See `calibration.yaml`
for the full derivation and current calibration status.

## Layout

```
Code/        Python scripts (pipeline + figure generators)
Docs/        Writeup PDF/source, setup notes
Data/        Trial data (gitignored except small legacy CSV)
figures/     Generated figures used by the writeup
```

Model checkpoints (`*.pth`, `*.pt`) and Python venvs (`.venv_*/`) live at
the repo root, are gitignored, and must be created locally.

## Setup

Three Python environments are used (versions and reasons documented in
`CLAUDE.md`):

- `.venv_analysis` — Python 3.9, pandas/numpy/matplotlib, plus `mcap`/
  `mcap-ros2-support` for the `.mcap` fallback extractor
- `.venv_sam2` — Python 3.11, SAM 2 + torch with MPS
- `.venv_dado` — Python 3.11, transformers (DINOv2 + Depth-Anything)

A separate conda environment (`occ`, pythonocc-core via conda-forge) is used
only for CAD/STEP-file inspection (`Code/cad_extract_transform.py`,
`Code/cad_find_lens_occ.py`) — not part of the regular per-bag pipeline.

Checkpoints:

- SAM 1 ViT-H: `sam_vit_h_4b8939.pth`
  (`https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`)
- SAM 2.1 Hiera-L: `sam2.1_hiera_large.pt`
  (`Code/download_sam2_ckpt.py` fetches it)

## End-to-end run on a bag

Run from the repo root. `<trial>` is a bag folder exported by the lab's
`ros2_unbag` pipeline (e.g. `Data/lfdws_t001/lfdws_t001`).

```bash
# 1. detect grasp / release / force-contact events on the merged CSV
.venv_analysis/bin/python Code/analyze_demo.py --trial <trial> --out <fig_dir>

# 2. convert PNGs to the zero-padded JPGs SAM 2 expects
.venv_sam2/bin/python Code/prepare_sam2_frames.py \
    --src <trial>/zed_zed_node_rgb_color_rect_image_compressed \
    --dst frames_jpg

# 3. auto-pick a SAM 2 seed point per role (no hard-coded image fractions)
.venv_sam2/bin/python Code/auto_seed.py --trial <trial> --ckpt sam_vit_h_4b8939.pth

# 4. propagate each tracked object across the demo, bidirectionally
# (add --offload_video_to_cpu if the trial has enough frames to exceed
#  device memory during init_state -- see CLAUDE.md)
.venv_sam2/bin/python Code/propagate_demo_bidir.py \
    --trial <trial> --ckpt sam2.1_hiera_large.pt --jpg_dir frames_jpg \
    --out <fig_dir>/propagation_grasped
# for additional objects beyond the first two roles, seed manually:
.venv_sam2/bin/python Code/propagate_object_n.py \
    --trial <trial> --ckpt sam2.1_hiera_large.pt --jpg_dir frames_jpg \
    --obj_id 3 --role <role_name> --seed_img_id <id> --seed_box "x0,y0,x1,y1" \
    --out <fig_dir>/propagation_obj3

# 5. compose the sidecar bundle from however many objects were propagated
.venv_analysis/bin/python Code/build_sidecar_multi.py \
    --trial <trial> \
    --object "1:grasped:<fig_dir>/propagation_grasped_summary.csv:0,255,0" \
    --object "2:contact_receiver:<fig_dir>/propagation_cup_summary.csv:255,0,255" \
    --out <fig_dir>/identify
```

`Code/build_sidecar_multi.py` is the canonical sidecar builder — it accepts
any number of `--object obj_id:role:summary_csv:bgr_color` entries, so a
single-object trial (no contact event) and a four-object trial both go
through the same tool. `Code/build_sidecar.py` (a fixed two-role version)
is kept for reference but superseded.

Output bundle: `objects.json`, `objects_summary.csv`, per-frame overlays,
and a stitched MP4. Use a distinct `--out` per trial so results don't
overwrite each other — every script that writes a shared output path also
backs up any pre-existing file to `<path>.bak` first, as a second layer of
protection.

Optional follow-ups:

```bash
.venv_analysis/bin/python Code/mask_area_plot.py --trial <trial>  # mask-area-over-time figure
.venv_analysis/bin/python Code/force_overlay.py --trial <trial>   # uncalibrated force-arrow sanity check
.venv_dado/bin/python Code/object_identity.py                     # stable identities per object_id, single trial
.venv_dado/bin/python Code/object_identity_cross_trial.py         # same, pooled across multiple trials
.venv_analysis/bin/python Code/trial_report.py --trial <trial> --sidecar_json <fig_dir>/identify/objects.json --fig_dir <fig_dir>   # one diagnostic PDF
.venv_analysis/bin/python Code/project_ee.py --trial <trial>      # EE / wrench-line projection
.venv_analysis/bin/python Code/calibrate_hand_eye.py solve --trial <calib_trial> --square_size_m <m> --marker_size_m <m>  # recover bota→camera by hand-eye calibration
```

`project_ee.py` reads `calibration.yaml` (camera intrinsics, the fixed
`bota→camera` bracket transform). Until both are marked `filled: true` it
runs in a DRY mode that reports the bota/base-frame geometry but draws
nothing, so no placeholder calibration is ever used. As of now, intrinsics
are real (lab-provided); `bota→camera` is a documented preliminary CAD
estimate that turned out not to be trustworthy (both candidate lens
positions project the wrench ray outside the image on every real trial
tested) — see `calibration.yaml` for the full derivation and open caveats.
`calibrate_hand_eye.py` recovers `bota→camera` by direct measurement
(ChArUco board + `cv2.calibrateHandEye`) instead of reading it off CAD,
and is the recommended path once rig access is available; it never writes
`calibration.yaml` automatically, only prints a result for manual review.
The Franka Research 3 arm URDF is vendored at `Data/fr3.urdf`.

Step 3 (`auto_seed.py`) is optional: if the seed CSV doesn't exist, the
propagation scripts fall back to hard-coded defaults tuned to the original
trial — expect those to need a manual reseed on a sufficiently different
scene (see `Code/auto_seed.py`'s docstring for the known failure mode).

If a bag arrives as a raw `.mcap` instead of a `ros2_unbag` export (e.g. the
lab's exporter doesn't yet handle a depth topic, or the bag hasn't been run
through the export pipeline at all), `Code/mcap_extract.py` produces the
same merged-CSV + PNG layout directly from the bag — no ROS 2 install
needed:

```bash
.venv_analysis/bin/python Code/mcap_extract.py \
    --bag <trial>_0.mcap --trial_name <trial> --out Data
```

## Writeup

```bash
pdflatex -output-directory=Docs Docs/writeup.tex
```

(Run twice to resolve references.) `Docs/writeup.tex` is a running progress
document — pages accumulate as dated update sections; earlier sections are
never edited, only appended to.

## More

- `CLAUDE.md` — full pipeline notes, hard-coded knobs, conventions.
- `Docs/setup_info.md` — legacy ROS 2 / bag-export setup notes.
- `Docs/publication.tex` — early-draft paper skeleton (separate from the
  writeup; not the lab deliverable).
