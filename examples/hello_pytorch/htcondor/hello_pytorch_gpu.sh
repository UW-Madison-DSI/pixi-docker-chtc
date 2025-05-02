#!/bin/bash
echo -e "# Hello CHTC from Job ${1} running on $(hostname)\n"

echo -e "# Check to see if the NVIDIA drivers can correctly detect the GPU:\n"
nvidia-smi

echo -e "# Check if PyTorch can detect the GPU:\n"
python ./src/torch_detect_GPU.py
