#!/usr/bin/env bash

# Single-node PyTorch distributed training with NCCL and torchrun
# Uses torchrun to spawn multiple processes (one per GPU) within a single SLURM task

#SBATCH --job-name="multi_gpu_single_node"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=4G  # memory per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # One task; torchrun spawns processes for each GPU
#SBATCH --cpus-per-task=2    # CPUs per task (used for DataLoader num_workers)
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=2    # Allocate 2 GPUs to the single task
#SBATCH --gpu-bind=none  # NCCL does not work with --gpu-bind (e.g. --gpu-bind=closest maps all devices to device ID 0)
#SBATCH --account=<ACCOUNT NAME>  # match to an "Account" returned by the 'accounts' command
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH --time=01:00:00
#SBATCH --error multi_gpu_single_node.slurm-%j.err
#SBATCH --output multi_gpu_single_node.slurm-%j.out

module reset

echo "job is starting on $(hostname)"

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=$((12000 + $SLURM_JOB_ID % 1000))

# Pass any training arguments to execute.sh (e.g., --amp --checkpoint-freq 5)
srun execute.sh "${TRAINING_ARGS:-}"
