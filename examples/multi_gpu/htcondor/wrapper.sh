#!/usr/bin/env bash

# detailed logging to stderr
set -x

echo -e "# Hello CHTC from Job ${1} running on $(hostname)\n"
echo -e "# GPUs assigned: ${CUDA_VISIBLE_DEVICES}\n"

echo -e "# Activate Pixi environment\n"
# The last line of the entrypoint.sh file is 'exec "$@"'. If this shell script
# receives arguments, exec will interpret them as arguments to it, which is not
# intended. To avoid this, strip the last line of entrypoint.sh and source that
# instead.
. <(sed '$d' /app/entrypoint.sh)

# Note: Use of nvidia-smi in Apptainer requires the '--nvccli' option.
# https://apptainer.org/docs/user/main/gpu.html#nvidia-gpus-cuda-nvidia-container-cli
# As of 2025-06-12, CHTC supports '--nv' but not '--nvccli' and so 'nvidia-smi'
# can not be used.
#
echo -e "# Check to see if the NVIDIA drivers can correctly detect the GPU:\n"
nvidia-smi

echo -e "\n# Check that the training code exists:\n"
ls -1ap ./src/

echo -e "\n# Train MNIST with PyTorch:\n"
time python ./src/wrapper_htcondor.py
