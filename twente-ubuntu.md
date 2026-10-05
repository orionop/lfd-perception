# Ubuntu RTX 4080 — Codex execution handoff

## Objective and constraints

Execute the already-decided interaction-model bakeoff for lab deliverable A,
entirely on this Ubuntu PC, including inference and evaluation. The operator
uses Codex on this PC. No Mac execution or return-transfer workflow is required.

Read `LAB_DELIVERABLE_A.md` and `Docs/MONDAY_INTERACTION_BAKEOFF.md` before
acting. They define the experiment. Calibration and manuscript B are archived.

Preserve these decisions:

- HOI-DETR evaluates five grasp cases and seven contact cases.
- DistinctNet evaluates the five grasp cases, separately for raw and stabilized
  frame pairs.
- Use the existing frozen benchmark, cached SAM proposals, adapters, manifest,
  thresholds, and evaluator.
- Do not regenerate inputs, tune regions/weights, change case timing, alter
  reference labels, weaken thresholds, or substitute models.
- Fix execution/setup defects only when needed to run this experiment; record
  each fix and its reason. Preserve user changes and existing results.
- Do not integrate any model into production during this experiment.

## 1. Clone this project

Only this project needs a manual clone. The runner clones HOI-DETR and
DistinctNet at the pinned commits and downloads checkpoints automatically.

Choose a working directory with sufficient disk space, then run:

```bash
git clone --branch interaction-bakeoff-prep --single-branch https://github.com/orionop/utwente.git
cd utwente
git status --short --branch
git log -5 --oneline
```

If the project already exists on this PC, use that checkout instead:

```bash
cd /absolute/path/to/utwente
git status --short --branch
git checkout interaction-bakeoff-prep
git pull --ff-only
git log -5 --oneline
```

Replace the example absolute path with the actual checkout. Do not discard
local changes to switch branches. The September 10 preparation commit is
`e2e6360`; the checkout must contain that preparation or a later descendant.
This handoff is newly created locally: ensure it reaches the Ubuntu checkout
by copying this file or committing/pushing it from its originating checkout.
Do not assume it is already on GitHub.

Check the tracked execution inputs:

```bash
test -f figures/interaction_bakeoff/input/benchmark.json
test -f config/evaluation_manifest.yaml
test -f scripts/run_interaction_bakeoff_gpu.sh
test -f docker/Dockerfile.distinctnet-bakeoff
test -f Code/run_hoi_detr_bakeoff.py
test -f Code/run_distinctnet_bakeoff.py
test -f Code/score_interaction_bakeoff.py
bash -n scripts/run_interaction_bakeoff_gpu.sh
```

## 2. Host prerequisites

Required: Ubuntu 22.04, RTX 4080, working NVIDIA driver, Docker, NVIDIA
Container Toolkit, Git, curl, internet access to GitHub/Hugging Face/Google
Drive/container registries, and approximately 25 GB free for images and model
weights **plus** space for evaluation data and outputs. No host SAM checkpoints,
Isaac Sim, or Mac virtual environments are needed for this bakeoff.

```bash
git --version
curl --version
docker --version
docker info
nvidia-smi
df -h .
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

The last command must show the RTX 4080 inside the container. Isaac Sim working
on the host does not by itself verify Docker GPU access.

If Git/curl are absent:

```bash
sudo apt-get update
sudo apt-get install -y git curl
```

If the Docker daemon is stopped:

```bash
sudo systemctl start docker
```

If Docker has a permissions error, configure access for the current user using
the machine's administrator-approved procedure, then reopen the terminal and
repeat `docker info`. Avoid running the whole experiment as root.

If Docker cannot select an NVIDIA runtime, inspect `nvidia-ctk --version` and
the installed NVIDIA Container Toolkit. If installed but unconfigured:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

If the toolkit is absent, use NVIDIA's current official Ubuntu installation
instructions to configure its package repository and install it. Do not assume
Ubuntu's default apt sources contain `nvidia-container-toolkit`.

## 3. Put evaluation data on this PC before the full run

The tracked frozen bundle is sufficient for inference. **Final scoring also
requires reference data outside that bundle.** A Git clone does not supply
all of it: many `Data/` recordings, reference CSVs, and overlay folders are
gitignored.

Use the existing lab share, backup, or complete project data copy to populate
the paths in `config/evaluation_manifest.yaml` on this PC. Do not recreate
ground truth or rerun SAM to replace the existing reference tracks.

Required reference CSVs:

```text
figures/identify/objects_summary.csv
figures/identify_depth_multi/objects_summary.csv
figures/t001labexport/identify/objects_summary.csv
figures/t002new/identify/objects_summary.csv
figures/t002labexport/identify/objects_summary.csv
figures/t004/identify/objects_summary.csv
figures/t005/identify/objects_summary.csv
```

The evaluator also needs the event RGB PNGs under each manifest recording's
`trial/zed_zed_node_rgb_color_rect_image_compressed/` directory, and overlays
referenced by the CSVs' `overlay_path` column. Preserve relative layout. Inspect
absolute paths for references to another machine; map them to the equivalent
existing local files without changing masks, roles, or case membership. Record
any path-only adaptation. Do not accept a bounding-box fallback or missing
reference as an equivalent replacement for the intended mask evaluation.

Codex must inspect `benchmark.json`, the manifest, and the reference CSVs and
verify that **every frozen case** has its intended RGB and reference overlay
available before expensive inference. If files are missing, report the exact
paths and obtain the existing files. Do not claim a complete scored experiment
with missing evidence.

## 4. Known execution issue to resolve before launching

The existing runner contains:

```bash
docker build --tag hoi-detr-bakeoff:2026-09-03 "${HOI_REPO}"
```

At the pinned HOI-DETR commit, the intended Dockerfile is
`docker/Dockerfile`, rather than a root `Dockerfile`. Confirm this after cloning
the pinned upstream repository. The minimal execution correction is to specify
that file while keeping the same build context:

```bash
docker build --file "${HOI_REPO}/docker/Dockerfile" \
  --tag hoi-detr-bakeoff:2026-09-03 "${HOI_REPO}"
```

This is a build-path correction, not a change to the model or experiment.
Codex should apply it on this PC and record the diff before launching. This
handoff does not itself modify the runner.

The upstream image uses an older PyTorch/CUDA/MMCV stack. RTX 4080 runtime
compatibility and external package downloads have not yet been proven by a
GPU run. Treat failures as environment failures; inspect the actual error,
preserve logs, and make only necessary compatible setup corrections. Do not
silently change checkpoints, inference parameters, benchmark inputs, or
selection rules to make execution succeed.

Pinned upstream commits:

```text
HOI-DETR:   1b367292f3833afd64a204bd4d9d84519541d035
DistinctNet: f4c2c05488216795e0c2ba39edaa8acbf60e4732
```

## 5. Run the decided experiment

Run from the project root. Keep this shell open; model downloads and image
builds can take substantial time. Install/use tmux if the terminal session is
likely to disconnect.

```bash
mkdir -p figures/interaction_bakeoff/run_logs
export BAKEOFF_RUN_ID=rtx4080_20260930_01
test ! -e "figures/interaction_bakeoff/gpu_outputs/${BAKEOFF_RUN_ID}"
set -o pipefail
bash scripts/run_interaction_bakeoff_gpu.sh 2>&1 | tee "figures/interaction_bakeoff/run_logs/${BAKEOFF_RUN_ID}.log"
```

If the existence check fails, choose a new run ID before starting. Do not
overwrite a previous run. The runner sequentially:

1. Clones/fetches the pinned external repositories.
2. Downloads published checkpoints into `.external/interaction_bakeoff/weights`.
3. Builds isolated HOI-DETR and DistinctNet containers.
4. Runs HOI-DETR on the twelve frozen event cases.
5. Runs DistinctNet on raw and stabilized grasp frame pairs.
6. Maps outputs to the cached SAM proposals and runs the existing evaluator.
7. Writes the final verdict without modifying the production selector.

Expected outputs:

```text
figures/interaction_bakeoff/gpu_outputs/<run-id>/hoi_detr/predictions.json
figures/interaction_bakeoff/gpu_outputs/<run-id>/distinctnet/predictions.json
figures/interaction_bakeoff/gpu_outputs/<run-id>/scored/verdict.json
```

Inspect the result:

```bash
cat "figures/interaction_bakeoff/gpu_outputs/${BAKEOFF_RUN_ID}/scored/verdict.json"
ls -lh "figures/interaction_bakeoff/gpu_outputs/${BAKEOFF_RUN_ID}/scored/"
git diff -- scripts/run_interaction_bakeoff_gpu.sh
```

Preserve the log, predictions, masks, per-case selection reports, evaluation
reports, benchmark hash, pinned commits, host GPU/driver details, and setup
changes. Outputs and checkpoints may be gitignored; keep them on this PC and
back up the run through the available lab storage.

## 6. Failure handling and retries

Read the first substantive error in the log. Distinguish build/download,
CUDA/runtime, missing-data, and scoring failures from model abstentions.

- A failed model load is not a 0/5 scientific result.
- Missing reference data is not an abstention or successful evaluation.
- If a download was interrupted, check the checkpoint: the runner skips any
  nonempty file, so a partial checkpoint can survive into retries. Resume or
  quarantine that exact partial file and download the same checkpoint again.
- If the runtime exhausts GPU memory, record the error and inspect the existing
  upstream/adapters before deciding an execution-only accommodation. Keep
  frozen input resolution and model settings unless a change is explicitly
  reviewed and recorded.
- Preserve failed run folders. Use a new run ID for a full retry:

```bash
export BAKEOFF_RUN_ID=rtx4080_20260930_retry_02
set -o pipefail
bash scripts/run_interaction_bakeoff_gpu.sh 2>&1 | tee "figures/interaction_bakeoff/run_logs/${BAKEOFF_RUN_ID}.log"
```

If inference completed but scoring failed, fix the data/setup issue and score
the preserved predictions into a **new** scoring directory instead of
re-running inference:

```bash
docker run --rm \
  --volume "$PWD:/work" \
  --workdir /work \
  distinctnet-bakeoff:2026-09-03 \
  python /work/Code/score_interaction_bakeoff.py \
    --bundle /work/figures/interaction_bakeoff/input \
    --hoi-predictions "/work/figures/interaction_bakeoff/gpu_outputs/${BAKEOFF_RUN_ID}/hoi_detr/predictions.json" \
    --distinct-predictions "/work/figures/interaction_bakeoff/gpu_outputs/${BAKEOFF_RUN_ID}/distinctnet/predictions.json" \
    --out "/work/figures/interaction_bakeoff/gpu_outputs/${BAKEOFF_RUN_ID}/scored_retry_02"
```

Use the run ID of the completed inference; choose an unused scoring directory.

## 7. Frozen interpretation and stopping rules

- Grasp passes only with at least **4/5 accepted, all correct**, across at
  least **three independent groups**.
- Contact passes only with at least **6/7 accepted, all correct**, across
  **all three groups**.
- HOI-DETR must form the complete relation chain required by the adapter.
  Its human-hand training means failure to detect a robot gripper is a known
  transfer risk.
- DistinctNet is grasp-only; raw and stabilized outcomes stay separate.
  A foreground-to-proposal IoU below **0.50** remains an abstention.
- Check the per-role evidence and missing-case fields as well as the final
  Boolean verdict. Never report a pass with incomplete reference evidence.
- If neither model passes grasp, stop integration and seek new recordings.
- If contact fails, keep the established 7/7 contact proposal recall and stop
  this model route for contact. Follow the existing decision to obtain new
  recordings or a different identity cue.
- Do not resume heuristic weight tuning or archived calibration/paper work.
- Even a passing bakeoff does not automatically complete A or authorize
  production integration. Report it for the next explicit integration decision;
  new-recording validation and the actual lab ROS 2 consumer remain required.

## Final report Codex should produce

Report the run ID, commands executed, setup corrections, inference completion,
role-wise accepted/correct counts, independent groups, missing evidence,
raw-versus-stabilized DistinctNet results, stop-gate outcome, and paths to the
saved artifacts. State the next step warranted by the measured result.
