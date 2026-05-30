# RLBD — Robot Learning from Demonstration

Vision module for the lab's ROS 2 LfD pipeline. Given a demonstration bag
(robot pose, gripper, force/torque, ZED RGB), it identifies the task-relevant
objects per interaction phase and writes a JSON sidecar consumable by
downstream LfD code.

## Pipeline

![Pipeline overview](figures/pipeline.png)

Proprioceptive event detection on the merged CSV identifies the grasp, press,
and release moments. Each event indexes the corresponding ZED frame, which
seeds SAM 2 (frozen) with a point prompt. Bidirectional video propagation
yields per-frame, role-tagged masks aggregated into a JSON sidecar.

## Layout

```
Code/        Python scripts (pipeline + figure generators)
Docs/        Writeup PDF/source, setup notes, failure-mode log
Data/        Trial data (gitignored except small legacy CSV)
figures/     Generated figures used by the writeup
```

Model checkpoints (`*.pth`, `*.pt`) and Python venvs (`.venv_*/`) live at
the repo root, are gitignored, and must be created locally.

## Setup

Three Python environments are used (versions and reasons documented in
`CLAUDE.md`):

- `.venv_analysis` — Python 3.9, pandas/numpy/matplotlib
- `.venv_sam2` — Python 3.11, SAM 2 + torch with MPS
- `.venv_dado` — Python 3.11, transformers (DINOv2 + Depth-Anything)

Checkpoints:

- SAM 1 ViT-H: `sam_vit_h_4b8939.pth`
  (`https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`)
- SAM 2.1 Hiera-L: `sam2.1_hiera_large.pt`
  (`Code/download_sam2_ckpt.py` fetches it)

## End-to-end run on a bag

Run from the repo root. `<trial>` is a bag folder exported by the lab's
`ros2_unbag` pipeline (e.g. `Data/lfdws_t001/lfdws_t001`).

```bash
# 1. detect grasp / press / release events on the merged CSV
.venv_analysis/bin/python Code/analyze_demo.py --trial <trial>

# 2. convert PNGs to the zero-padded JPGs SAM 2 expects
.venv_sam2/bin/python Code/prepare_sam2_frames.py \
    --src <trial>/zed_zed_node_rgb_color_rect_image_compressed \
    --dst frames_jpg

# 3. auto-pick SAM 2 seed points (no hard-coded image fractions)
.venv_sam2/bin/python Code/auto_seed.py --trial <trial> --ckpt sam_vit_h_4b8939.pth

# 4. propagate both objects across the demo
.venv_sam2/bin/python Code/propagate_demo.py \
    --trial <trial> --ckpt sam2.1_hiera_large.pt --jpg_dir frames_jpg
.venv_sam2/bin/python Code/propagate_cup.py \
    --trial <trial> --ckpt sam2.1_hiera_large.pt --jpg_dir frames_jpg

# 5. compose the per-trial sidecar bundle
.venv_analysis/bin/python Code/build_sidecar.py
```

Output bundle in `figures/identify/`: `objects.json`, `objects_summary.csv`,
per-frame overlays, and a stitched MP4.

Optional follow-ups:

```bash
.venv_dado/bin/python Code/object_identity.py        # stable identities per object_id
.venv_analysis/bin/python Code/trial_report.py --trial <trial>   # one diagnostic PDF
```

Step 3 (`auto_seed.py`) is optional: if `figures/identify/auto_seeds.csv`
doesn't exist, the propagation scripts fall back to the hard-coded defaults
in `propagate_demo.py` / `propagate_cup.py`.

## Writeup

```bash
pdflatex -output-directory=Docs Docs/writeup.tex
```

(Run twice to resolve references.)

## More

- `CLAUDE.md` — full pipeline notes, hard-coded knobs, conventions.
- `Docs/FAILURE_MODES.md` — what does not work yet on the current trial.
- `Docs/setup_info.md` — legacy ROS 2 / bag-export setup notes.
