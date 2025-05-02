#!/bin/bash
echo -e "# Hello CHTC from Job ${1} running on $(hostname)\n"

echo -e "# Activating Pixi environment\n"
. /app/entrypoint.sh

echo -e "# Check to see if the NVIDIA drivers can correctly detect the GPU:\n"
nvidia-smi

echo -e "# Check if PyTorch can detect the GPU:\n"
python /app/src/torch_detect_GPU.py
