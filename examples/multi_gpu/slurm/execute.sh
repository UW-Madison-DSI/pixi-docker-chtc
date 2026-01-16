#!/usr/bin/env bash

# detailed logging to stderr
set -x

echo -e "# Hello CHTC from Job ${SLURM_JOB_ID} running on $(hostname)\n"
echo -e "# GPUs assigned: ${CUDA_VISIBLE_DEVICES}\n"

echo -e "# Check to see if the NVIDIA drivers can correctly detect the GPU:\n"
nvidia-smi

nvidia-smi --query-gpu=name,compute_cap

if [ -z "${SLURM_GPUS_PER_NODE}" ] || [ "${SLURM_GPUS_PER_NODE}" -le 0 ]; then
    echo "Error: SLURM_GPUS_PER_NODE is not set or is an invalid number."
    exit 1
fi

# Print job information
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on nodes: ${SLURM_NODELIST}"
echo "Number of tasks: ${SLURM_NTASKS}"
echo "GPUs per node: ${SLURM_GPUS_PER_NODE}"
echo "Master address: ${MASTER_ADDR}"
echo "Master port: ${MASTER_PORT}"
echo ""

echo -e "\n# Check that the training code exists:\n"
ls -1ap ../src/

echo -e "\n# Train MNIST with PyTorch using torchrun:\n"
echo "Using torchrun for distributed training"
echo "  - nproc-per-node: ${SLURM_GPUS_PER_NODE} (matching --gpus-per-node)"
echo "  - nnodes: ${SLURM_NNODES} (from SLURM)"
echo "  - node-rank: ${SLURM_NODEID} (from SLURM)"
echo ""

# torchrun with SLURM integration
# - nproc-per-node: number of processes per node (should match GPUs per node)
# - nnodes: total number of nodes (from SLURM_NNODES)
# - node-rank: rank of this node (from SLURM_NODEID)
# - rdzv-backend: rendezvous backend (c10d is standard for static clusters)
# - rdzv-endpoint: master node address and port (already set as env vars)
#
# Training script arguments (passed after --):
# --amp: Enable automatic mixed precision for faster training on modern GPUs
# --checkpoint-freq: Save checkpoint every N epochs (0 to disable)

# Filter out empty arguments to avoid passing "" to the training script
TRAINING_ARGS=()
for arg in "$@"; do
    if [ -n "$arg" ]; then
        TRAINING_ARGS+=("$arg")
    fi
done

time pixi run --environment gpu torchrun \
    --nproc-per-node="${SLURM_GPUS_PER_NODE}" \
    --nnodes="${SLURM_NNODES}" \
    --node-rank="${SLURM_NODEID}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    ../src/torch_mnist_multi_gpu.py "${TRAINING_ARGS[@]}"
