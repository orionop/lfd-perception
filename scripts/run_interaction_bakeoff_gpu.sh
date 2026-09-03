#!/usr/bin/env bash
set -euo pipefail

# Run from a complete copy of the utwente repository on an NVIDIA Docker host.
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_STAGE_ROOT="${BAKEOFF_GPU_ROOT:-${TASK_ROOT}/.external/interaction_bakeoff}"
RUN_ID="${BAKEOFF_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${TASK_ROOT}/figures/interaction_bakeoff/gpu_outputs/${RUN_ID}"
BUNDLE="${TASK_ROOT}/figures/interaction_bakeoff/input"

HOI_COMMIT="1b367292f3833afd64a204bd4d9d84519541d035"
DISTINCT_COMMIT="f4c2c05488216795e0c2ba39edaa8acbf60e4732"
HOI_REPO="${GPU_STAGE_ROOT}/HOI-DETR"
DISTINCT_REPO="${GPU_STAGE_ROOT}/DistinctNet"
WEIGHT_ROOT="${GPU_STAGE_ROOT}/weights"
HOI_CHECKPOINT="${WEIGHT_ROOT}/hoi_detr/epoch_5.pth"
DISTINCT_RUN="${WEIGHT_ROOT}/distinctnet/motion_run"
DISTINCT_CHECKPOINT="${DISTINCT_RUN}/models/model.pth"

command -v docker >/dev/null
nvidia-smi >/dev/null
docker info >/dev/null
test -f "${BUNDLE}/benchmark.json"
test ! -e "${OUTPUT_ROOT}"

mkdir -p "${GPU_STAGE_ROOT}" "${WEIGHT_ROOT}/hoi_detr" \
         "${DISTINCT_RUN}/models" "${OUTPUT_ROOT}"

if [[ ! -d "${HOI_REPO}/.git" ]]; then
  git clone https://github.com/AhmadDarKhalil/HOI-DETR.git "${HOI_REPO}"
fi
git -C "${HOI_REPO}" fetch --depth 1 origin "${HOI_COMMIT}"
git -C "${HOI_REPO}" checkout --detach "${HOI_COMMIT}"

if [[ ! -d "${DISTINCT_REPO}/.git" ]]; then
  git clone https://github.com/DLR-RM/DistinctNet.git "${DISTINCT_REPO}"
fi
git -C "${DISTINCT_REPO}" fetch --depth 1 origin "${DISTINCT_COMMIT}"
git -C "${DISTINCT_REPO}" checkout --detach "${DISTINCT_COMMIT}"

if [[ ! -s "${HOI_CHECKPOINT}" ]]; then
  curl --fail --location \
    https://huggingface.co/ahmaddarkhalil/hoi-detr/resolve/main/epoch_5.pth \
    --output "${HOI_CHECKPOINT}"
fi

docker build --tag hoi-detr-bakeoff:2026-09-03 "${HOI_REPO}"
docker build --file "${TASK_ROOT}/docker/Dockerfile.distinctnet-bakeoff" \
  --tag distinctnet-bakeoff:2026-09-03 "${TASK_ROOT}"

if [[ ! -s "${DISTINCT_CHECKPOINT}" ]]; then
  cp "${DISTINCT_REPO}/configs/motion.yaml" "${DISTINCT_RUN}/config.yaml"
  docker run --rm \
    --volume "${WEIGHT_ROOT}:/weights" \
    distinctnet-bakeoff:2026-09-03 \
    gdown 1tWoSG8wyHqZ2kZQNgyb9KaOTioW5kc9w \
      --output /weights/distinctnet/motion_run/models/model.pth
fi

docker run --rm --gpus all \
  --volume "${HOI_REPO}:/external/HOI-DETR" \
  --volume "${TASK_ROOT}:/work" \
  --volume "${WEIGHT_ROOT}:/weights:ro" \
  --workdir /external/HOI-DETR \
  hoi-detr-bakeoff:2026-09-03 \
  python /work/Code/run_hoi_detr_bakeoff.py \
    --repo /external/HOI-DETR \
    --bundle /work/figures/interaction_bakeoff/input \
    --checkpoint /weights/hoi_detr/epoch_5.pth \
    --out "/work/figures/interaction_bakeoff/gpu_outputs/${RUN_ID}/hoi_detr"

docker run --rm --gpus all \
  --volume "${DISTINCT_REPO}:/external/DistinctNet:ro" \
  --volume "${TASK_ROOT}:/work" \
  --volume "${WEIGHT_ROOT}:/weights:ro" \
  --workdir /work \
  distinctnet-bakeoff:2026-09-03 \
  python /work/Code/run_distinctnet_bakeoff.py \
    --repo /external/DistinctNet \
    --bundle /work/figures/interaction_bakeoff/input \
    --checkpoint /weights/distinctnet/motion_run/models/model.pth \
    --out "/work/figures/interaction_bakeoff/gpu_outputs/${RUN_ID}/distinctnet"

docker run --rm \
  --volume "${TASK_ROOT}:/work" \
  --workdir /work \
  distinctnet-bakeoff:2026-09-03 \
  python /work/Code/score_interaction_bakeoff.py \
    --bundle /work/figures/interaction_bakeoff/input \
    --hoi-predictions "/work/figures/interaction_bakeoff/gpu_outputs/${RUN_ID}/hoi_detr/predictions.json" \
    --distinct-predictions "/work/figures/interaction_bakeoff/gpu_outputs/${RUN_ID}/distinctnet/predictions.json" \
    --out "/work/figures/interaction_bakeoff/gpu_outputs/${RUN_ID}/scored"

echo "Run complete: ${OUTPUT_ROOT}/scored/verdict.json"
