#!/usr/bin/env bash

# Ensure existing models are backed up
if [ -f "mnist_cnn.pt" ]; then
    mv mnist_cnn.pt mnist_cnn_"$(date '+%Y-%m-%d-%H-%M')".pt.bak
fi

# sbatch multi_gpu_single_node.sh
sbatch multi_gpu_multi_node.sh

# Training script arguments can be passed via SLURM environment variables.
# Example with mixed precision and checkpointing:
#   TRAINING_ARGS="--amp --checkpoint-freq 5" sbatch multi_gpu_multi_node.sh
#
# Available training arguments:
#   --batch-size N      Batch size per GPU (default: 64)
#   --epochs N          Number of epochs (default: 20)
#   --lr LR             Learning rate (default: 1.0)
#   --amp               Enable automatic mixed precision
#   --checkpoint-freq N Save checkpoint every N epochs (0 to disable)
#   --resume PATH       Resume from checkpoint file
