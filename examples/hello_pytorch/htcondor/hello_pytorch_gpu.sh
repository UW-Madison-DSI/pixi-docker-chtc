#!/bin/bash
echo -e "# Hello CHTC from Job ${1} running on $(hostname)\n"

echo -e "# Activating Pixi environment\n"
# The last line of the entrypoint.sh file is 'exec "$@"'. If this shell script
# receives arguments, exec will interpret them as arguments to it, which is not
# intended. To avoid this, strip the last line of entrypoint.sh and source that
# instead.
. <(sed '$d' /app/entrypoint.sh)

echo -e "# Check to see if the NVIDIA drivers can correctly detect the GPU:\n"
nvidia-smi

echo -e "\n# Check if PyTorch can detect the GPU:\n"
python /app/src/torch_detect_GPU.py
