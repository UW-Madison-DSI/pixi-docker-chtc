#!/usr/bin/env bash

# Ensure existing models are backed up
if [ -f "mnist_cnn.pt" ]; then
    mv mnist_cnn.pt mnist_cnn_"$(date '+%Y-%m-%d-%H-%M')".pt.bak
fi

# sbatch multi_gpu_single_node.sh
sbatch multi_gpu_multi_node.sh
