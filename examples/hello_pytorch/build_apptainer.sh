#!/bin/bash

_sha=$(git rev-parse --verify --short HEAD)
apptainer build pixi-docker-chtc-sha-"${_sha}".sif apptainer.def

# apptainer run --nv --containall pixi-docker-chtc*.sif bash

# Note: The %runscript in the definition file is used to pass commands into the environment
# apptainer run --nv --containall pixi-docker-chtc*.sif python /app/src/torch_detect_GPU.py

# Note: 'apptainer exec' ignores the %runscript and the %startscript from the definition file
# apptainer exec --nv --containall pixi-docker-chtc*.sif /app/entrypoint.sh python /app/src/torch_detect_GPU.py
