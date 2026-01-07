#!/usr/bin/env bash

#SBATCH --job-name="multi_gpu"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=2
#SBATCH --gpu-bind=none  # NCCL does not work with --gpu-bind (e.g. --gpu-bind=closest maps all devices to device ID 0)
#SBATCH --account=<ACCOUNT NAME>  # match to an "Account" returned by the 'accounts' command
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH --time=01:00:00
#SBATCH --error multi_gpu.slurm-%j.err
#SBATCH --output multi_gpu.slurm-%j.out

module reset

echo "job is starting on $(hostname)"

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=$((12000 + $SLURM_JOB_ID % 1000))

srun execute.sh
