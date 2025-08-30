#!/bin/bash

# detailed logging to stderr
set -x

echo -e "# Hello CHTC from Job ${1} running on $(hostname)\n"
echo -e "# GPUs assigned: ${CUDA_VISIBLE_DEVICES}\n"

echo -e "# Check to see if the NVIDIA drivers can correctly detect the GPU:\n"
nvidia-smi

nvidia-smi --query-gpu=name,compute_cap

echo -e "\n# Check if PyTorch can detect the GPU:\n"
pixi run --environment gpu python ../src/torch_detect_GPU.py

# echo -e "\n# Extract the training data:\n"
# if [ -f "MNIST_data.tar.gz" ]; then
#     tar -vxzf MNIST_data.tar.gz
# else
#     echo "The training data archive, MNIST_data.tar.gz, is not found."
#     echo "Please transfer it to the worker node in the HTCondor jobs submission file."
#     exit 1
# fi

echo -e "\n# Check that the training code exists:\n"
ls -1ap ../src/

echo -e "\n# Train MNIST with PyTorch:\n"
time pixi run --environment gpu python ../src/torch_MNIST.py --epochs 14 --data-dir ./data --save-model
