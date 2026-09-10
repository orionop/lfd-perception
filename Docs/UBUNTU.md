# RTX 4080 Ubuntu host — setup and preflight for the interaction bakeoff

One-time setup on the Ubuntu box before running
`scripts/run_interaction_bakeoff_gpu.sh`. Run every check in order — each one
catches a failure mode that would otherwise surface late, mid-script, after
it's already downloaded gigabytes.

## 1. Docker itself

```bash
docker --version
docker info >/dev/null && echo "docker daemon OK"
```

If `docker info` fails, the daemon isn't running — start it
(`sudo systemctl start docker`) before continuing.

## 2. NVIDIA driver sees the GPU

```bash
nvidia-smi
```

Must show the RTX 4080 and a driver version. If this fails, the NVIDIA
driver itself isn't installed/loaded — fix this first, nothing below will
work without it.

## 3. Docker can actually reach the GPU (the step that silently bites people)

Having Docker and the NVIDIA driver installed **separately** does not give
Docker GPU access on its own. This needs the **NVIDIA Container Toolkit**
installed and configured:

```bash
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

This must print the same GPU info as step 2, from *inside* a container. If
it fails with something like "could not select device driver" or "unknown
runtime", the toolkit is missing or unconfigured:

```bash
# install (Ubuntu, if missing)
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
# configure docker to use it
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Re-run the `docker run --rm --gpus all ...` check above until it passes
before moving on.

## 4. Disk space

```bash
df -h .
```

Need ~25GB free: HOI-DETR's checkpoint alone is 5.85GB, plus two Docker
images (one is a full CUDA devel base), plus DistinctNet's checkpoint.

## 5. Clone the repo — the right branch, not `main`

Everything this run needs (the frozen benchmark bundle, the adapters, the
Dockerfiles) lives on `interaction-bakeoff-prep`, **not yet merged into
`main`**. Cloning without checking this branch out gets an empty shell that
will fail immediately (no `figures/interaction_bakeoff/input/`).

```bash
git clone https://github.com/orionop/lfd-perception.git
cd lfd-perception
git checkout interaction-bakeoff-prep
```

Confirm the bundle actually came through:

```bash
test -f figures/interaction_bakeoff/input/benchmark.json && echo "bundle present"
```

Nothing else needs to be copied onto this box by hand — no raw `Data/`
recordings, no pre-cloned model repos, no checkpoints. The script clones
HOI-DETR and DistinctNet itself (pinned to exact commits) and downloads both
checkpoints automatically on first run.

## 6. Run the bakeoff

```bash
bash scripts/run_interaction_bakeoff_gpu.sh
```

Then read the result:

```bash
cat figures/interaction_bakeoff/gpu_outputs/*/scored/verdict.json
```

Stop gate (from `LAB_DELIVERABLE_A.md`): grasp needs ≥4/5 correct across
≥3 independent groups; contact needs ≥6/7 correct across all 3 groups. No
result auto-authorizes production integration regardless of outcome — see
`Docs/MONDAY_INTERACTION_BAKEOFF.md` for what to do with either result.
