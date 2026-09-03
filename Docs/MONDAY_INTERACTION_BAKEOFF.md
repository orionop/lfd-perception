# Monday interaction-model bakeoff

## Before Monday

The frozen input bundle is `figures/interaction_bakeoff/input` (21 MB): five
grasp cases, seven contact cases, their event images, and cached SAM proposals.
It contains no reference masks.  Do not regenerate it after seeing model
results.

Copy the complete repository to the RTX 4080 host.  The host needs Linux,
Docker with the NVIDIA Container Toolkit, `git`, `curl`, and enough free space
for two images and checkpoints (allow 25 GB).

## Execute

From the repository root:

```bash
bash scripts/run_interaction_bakeoff_gpu.sh
```

The script pins both external repositories, downloads their published
checkpoints, builds isolated containers, runs inference, maps predictions to
the frozen SAM proposals, and evaluates the results locally.  It never edits
the production selector or existing evaluation reports.

To resume a failed attempt without overwriting partial evidence, choose a new
run identifier:

```bash
BAKEOFF_RUN_ID=monday_retry_01 bash scripts/run_interaction_bakeoff_gpu.sh
```

## Read the result

Open:

```text
figures/interaction_bakeoff/gpu_outputs/<run-id>/scored/verdict.json
```

- HOI-DETR must pass grasp and contact under the frozen evaluator.
- DistinctNet is evaluated for grasp only, separately for raw and stabilized
  frame pairs.
- A missing relation or weak DistinctNet foreground match is an abstention.
- No result authorizes production integration automatically.

If neither model passes grasp, stop and wait for new recordings.  If contact
fails, retain the known 7/7 proposal result and wait for calibrated geometric
evidence.  Do not tune the fixed regions or confidence rules on these cases.
