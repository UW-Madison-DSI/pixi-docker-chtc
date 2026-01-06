#!/usr/bin/env bash

# detailed logging to stderr
set -x

echo -e "# Hello CHTC from Job ${1} running on $(hostname)\n"
echo -e "# GPUs assigned: ${CUDA_VISIBLE_DEVICES}\n"

echo -e "# Check to see if the NVIDIA drivers can correctly detect the GPU:\n"
nvidia-smi

nvidia-smi --query-gpu=name,compute_cap

# Print job information
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on nodes: ${SLURM_NODELIST}"
echo "Number of tasks: ${SLURM_NTASKS}"
echo "GPUs per node: 2"
echo "Master address: ${MASTER_ADDR}"
echo "Master port: ${MASTER_PORT}"
echo ""

echo -e "\n# Check that the training code exists:\n"
ls -1ap ../src/

echo -e "\n# Train MNIST with PyTorch:\n"
time pixi run --environment gpu python ../src/torch_mnist_multi_gpu.py
# time pixi run --environment gpu python ../src/torch_MNIST.py --epochs 14 --data-dir ./data --save-model
