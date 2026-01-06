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
        # Fallback: check CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible:
            gpu_list = [g.strip() for g in cuda_visible.split(",") if g.strip()]
            world_size = len(gpu_list) if gpu_list else 2
            print(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
        else:
            world_size = 2
            print(
                f"Warning: No GPU environment variables set, assuming {world_size} GPUs"
            )

    # Set master address and port for process group
    master_port = 12355

    print(f"Launching {world_size} training processes...")
    print(f"Master port: {master_port}")

    # Pre-download MNIST dataset before launching processes
    print("\nPre-downloading MNIST dataset to avoid race conditions...")
    try:
        import torch
        from torchvision import datasets, transforms

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        datasets.MNIST("./data", train=True, download=True, transform=transform)
        datasets.MNIST("./data", train=False, download=True, transform=transform)
        print("Dataset download complete!\n")
    except Exception as e:
        print(f"Warning: Could not pre-download dataset: {e}")
        print("Processes will attempt to download individually.\n")

    # Training script is colocated with wrapper script
    source_file_directory = Path(__file__).resolve().parent

    processes = []
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
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        processes.append(proc)

        # Small delay between launching processes
        time.sleep(2)

    print(f"\nAll {world_size} processes launched. Monitoring output...")

    # Monitor process output
    import select

    while processes:
        for proc in processes[:]:
            line = proc.stdout.readline()
            if line:
                rank_id = processes.index(proc)
                print(f"[rank{rank_id}]: {line.rstrip()}")

            # Check if process finished
            if proc.poll() is not None:
                # Read any remaining output
                for line in proc.stdout:
                    rank_id = processes.index(proc)
                    print(f"[rank{rank_id}]: {line.rstrip()}")
                processes.remove(proc)

        time.sleep(0.1)

    # Wait for all processes to complete and collect exit codes
    exit_codes = [proc.wait() for proc in processes]

    for i, code in enumerate(exit_codes):
        print(f"Process {i} finished with exit code {code}")

    # Check if all processes succeeded
    if all(code == 0 for code in exit_codes):
        print("\nTraining completed successfully!")
        return 0
    else:
        print(f"\nTraining failed! Exit codes: {exit_codes}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
