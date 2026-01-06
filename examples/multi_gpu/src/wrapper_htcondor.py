#!/usr/bin/env python3
"""
Wrapper script for launching multi-GPU PyTorch training on HTCondor.
This script spawns multiple processes for distributed training.
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def main():
    # Get number of GPUs from environment or command line
    # HTCondor sets _CONDOR_AssignedGPUs
    assigned_gpus = os.environ.get("_CONDOR_AssignedGPUs", "")

    if assigned_gpus:
        # Parse the assigned GPUs (format: "CUDA0, CUDA1" or similar)
        gpu_list = [g.strip() for g in assigned_gpus.split(",")]
        world_size = len(gpu_list)
        print(f"HTCondor assigned GPUs: {assigned_gpus}")
    else:
        # Fallback: assume 2 GPUs if not specified
        world_size = 2
        print(f"Warning: _CONDOR_AssignedGPUs not set, assuming {world_size} GPUs")

    # Set master address and port for process group
    master_port = 12355

    print(f"Launching {world_size} training processes...")
    print(f"Master port: {master_port}")

    processes = []

    # Training script is colocated with wrapper script
    source_file_directory = Path(__file__).resolve().parent

    # Launch a process for each GPU
    for rank in range(world_size):
        # Each process gets its rank and world_size as arguments
        cmd = [
            sys.executable,  # Use the same Python interpreter
            str(source_file_directory / "torch_mnist_multi_gpu_htcondor.py"),
            str(rank),
            str(world_size),
        ]

        print(f"Starting process {rank}/{world_size}...")

        # Launch subprocess
        proc = subprocess.Popen(cmd)
        processes.append(proc)

        # Small delay between launching processes
        time.sleep(1)

    print(f"\nAll {world_size} processes launched. Waiting for completion...")

    # Wait for all processes to complete
    exit_codes = []
    for i, proc in enumerate(processes):
        exit_code = proc.wait()
        exit_codes.append(exit_code)
        print(f"Process {i} finished with exit code {exit_code}")

    # Check if all processes succeeded
    if all(code == 0 for code in exit_codes):
        print("\nTraining completed successfully!")
        return 0
    else:
        print(f"\nTraining failed! Exit codes: {exit_codes}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
