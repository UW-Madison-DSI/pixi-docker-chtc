# Running the Hello PyTorch example on CHTC

This example contains two files:

* `hello_pytorch_gpu.sub`: The HTCondor job submission file
* `hello_pytorch_gpu.sh`: The execute shells script that contains the commands that are run as the job

## Running

Example of job output in the resulting `hello_pytorch_gpu.out.txt`

```
# Hello CHTC from Job 0 running on gpu2000.chtc.wisc.edu

# Activate Pixi environment

# Check to see if the NVIDIA drivers can correctly detect the GPU:

Fri May  2 18:11:58 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla P100-PCIE-16GB           On  |   00000000:3B:00.0 Off |                    0 |
| N/A   27C    P0             25W /  250W |       3MiB /  16384MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

# Check if PyTorch can detect the GPU:

PyTorch build CUDA version: 12.6
PyTorch build cuDNN version: 90800
PyTorch build NCCL version: (2, 26, 2)

Number of GPUs found on system: 1

Active GPU index: 0
Active GPU name: Tesla P100-PCIE-16GB
```
