#!/bin/bash

apptainer build uw-madison-dsi_pixi-docker-chtc_hello-pytorch-noble-cuda-12.9.sif apptainer.def

# apptainer run --nv --containall uw-madison-dsi_pixi-docker-chtc_hello-pytorch-noble-cuda-12.9.sif bash

# Note: The %runscript in the definition file is used to pass commands into the environment
# apptainer run --nv --containall uw-madison-dsi_pixi-docker-chtc_hello-pytorch-noble-cuda-12.9.sif python /app/src/torch_detect_GPU.py

# Note: 'apptainer exec' ignores the %runscript and the %startscript from the definition file
# apptainer exec --nv --containall uw-madison-dsi_pixi-docker-chtc_hello-pytorch-noble-cuda-12.9.sif /app/entrypoint.sh python /app/src/torch_detect_GPU.py
