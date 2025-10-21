#!/usr/bin/env bash
#SBATCH --job-name="hello_pytorch"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=2G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest  # select a cpu close to gpu on pci bus topology
#SBATCH --account=bexo-delta-gpu  # match to a "Project" returned by the "accounts" command
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH --time=01:00:00
#SBATCH --error hello_pytorch.slurm-%j.err
#SBATCH --output hello_pytorch.slurm-%j.out

module reset

echo "job is starting on $(hostname)"

nvidia-smi

srun execute.sh
