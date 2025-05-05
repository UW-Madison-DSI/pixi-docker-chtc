#!/bin/bash

# detailed logging to stderr
set -x

echo -e "# Hello CHTC from Job ${1} running on $(hostname)\n"
echo -e "# GPUs assigned: ${CUDA_VISIBLE_DEVICES}\n"

echo -e "# Unpack and install pixi-pack environment\n"
./environment.sh

echo -e "# Activate environment\n"
. activate.sh

echo -e "# Check to see if the NVIDIA drivers can correctly detect the GPU:\n"
nvidia-smi

echo -e "\n# Check if PyTorch can detect the GPU:\n"
python torch_detect_GPU.py

echo -e "\n# Extract the training data:\n"
if [ -f "MNIST_data.tar.gz" ]; then
    tar -vxzf MNIST_data.tar.gz
else
    echo "The training data archive, MNIST_data.tar.gz, is not found."
    echo "Please transfer it to the worker node in the HTCondor jobs submission file."
    exit 1
fi

echo -e "\n# Train MNIST with PyTorch:\n"
mkdir -p run
cd run
time python ../torch_MNIST.py --epochs 14 --save-model
