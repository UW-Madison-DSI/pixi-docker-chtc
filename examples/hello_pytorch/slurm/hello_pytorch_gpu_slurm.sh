#!/bin/bash
#SBATCH --job-name="hello_pytorch"
#SBATCH --output="hello_pytorch.out.%j.%N.out"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest  # select a cpu close to gpu on pci bus topology
#SBATCH --account=bexo-delta-gpu  # match to a "Project" returned by the "accounts" command
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 04:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

module reset

echo "job is starting on $(hostname)"

nvidia-smi

srun execute.sh
