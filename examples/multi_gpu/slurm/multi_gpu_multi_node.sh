#!/usr/bin/env bash

# Multi-node PyTorch distributed training with NCCL and torchrun
#
# This script uses torchrun (PyTorch's distributed launcher) for better
# portability and standardization compared to direct Python execution.
#
# Network requirements:
# - All nodes must be on the same network and able to communicate
# - Master node hostname must be resolvable by all nodes
# - Firewall must allow TCP communication on MASTER_PORT between nodes
# - For NCCL: InfiniBand or high-speed interconnect recommended for performance
#
# torchrun advantages:
# - Standardized PyTorch distributed training (scheduler-agnostic)
# - Better error handling and logging
# - Portable: works locally and on different HPC systems
# - Elastic training support (fault tolerance)

#SBATCH --job-name="multi_gpu_multi_node"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=4G  # memory per node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1  # The number of srun commands submitted per execute.sh script. Keep as 1.
#SBATCH --cpus-per-task=2  # Used to set num_workers in DataLoader
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=none  # NCCL does not work with --gpu-bind (e.g. --gpu-bind=closest maps all devices to device ID 0)
#SBATCH --account=<ACCOUNT NAME>  # match to an "Account" returned by the 'accounts' command
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH --time=01:00:00
#SBATCH --error multi_gpu_multi_node.slurm-%j.err
#SBATCH --output multi_gpu_multi_node.slurm-%j.out

module reset

echo "job is starting on $(hostname)"

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=$((12000 + $SLURM_JOB_ID % 1000))

srun execute.sh
