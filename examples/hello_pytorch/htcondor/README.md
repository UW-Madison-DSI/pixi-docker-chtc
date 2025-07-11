# Running the Hello PyTorch example on CHTC GPUs with Linux containers

This example contains the files:

* `hello_pytorch_gpu.sub`: The HTCondor job submission file
* `hello_pytorch_gpu.sh`: The execute shells script that contains the commands that are run as the job
* `submit.sh`: The shell script that submits `hello_pytorch_gpu.sub` to HTCondor as a standard batch job with `condor_submit`
* `interact.sh`: The shell script that submits `hello_pytorch_gpu.sub` to HTCondor as an interactive job with `condor_submit -interactive`

## Local testing

To test that your Linux container environment works as expected in advance of running on CHTC, you can run your container locally using a command like

```
docker run \
    --rm -ti \
    --gpus all \
    --user 12345 \
    -v $(pwd):/scratch \
    -w /scratch \
    <container image name:tag> \
    bash
```

where the reasoning for the options is:

* `--rm -ti`: Run the container interactively and clean it up on exit
* `--gpus all`: Indicates that GPUs that are available should be exposed
* `--user 12345`: Runs as a user that does not exist in `/etc/passwd` and so will default to a nameless user with the same permissions as a CHTC job
* `-v $(pwd):/scratch -w /scratch`: Mounts the current working directory to `/scratch` in the container filesystem and sets the container working directory there as well

### Example using the example's Docker container image

```
docker run --rm -ti --gpus all --user 12345 -v $(pwd):/scratch -w /scratch ghcr.io/uw-madison-dsi/pixi-docker-chtc:hello-pytorch-noble-cuda-12.9 bash
```

## Running

After logging onto CHTC and cloning this repository, navigate to this directory and run

```
bash submit.sh
```

This will submit the job and create the file `hello_pytorch_gpu.log.txt` which will have the job's log file written into it.
After the job finishes, it additionally return the job's `stdout` and `stderr` files:

* `docker_stderror`
* `hello_pytorch_gpu.err.txt`
* `hello_pytorch_gpu.out.txt`

### Example of job output in the resulting `hello_pytorch_gpu.out.txt`

```
# Hello CHTC from Job 0 running on gpu2000.chtc.wisc.edu

# GPUs assigned: GPU-02b71a7d

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

# Extract the training data:

data/
data/MNIST/
data/MNIST/raw/
data/MNIST/raw/train-images-idx3-ubyte.gz
data/MNIST/raw/train-images-idx3-ubyte
data/MNIST/raw/train-labels-idx1-ubyte.gz
data/MNIST/raw/train-labels-idx1-ubyte
data/MNIST/raw/t10k-images-idx3-ubyte.gz
data/MNIST/raw/t10k-images-idx3-ubyte
data/MNIST/raw/t10k-labels-idx1-ubyte.gz
data/MNIST/raw/t10k-labels-idx1-ubyte
data/MNIST/processed/
data/MNIST/processed/training.pt
data/MNIST/processed/test.pt

# Train MNIST with PyTorch:
...
Train Epoch: 14 [57600/60000 (96%)]	Loss: 0.000494
Train Epoch: 14 [58240/60000 (97%)]	Loss: 0.012554
Train Epoch: 14 [58880/60000 (98%)]	Loss: 0.016642
Train Epoch: 14 [59520/60000 (99%)]	Loss: 0.001183

Test set: Average loss: 0.0261, Accuracy: 9915/10000 (99%)

```

The job will also return the trained model serialized as a PyTorch model named `mnist_cnn.pt`.

### Resources

CHTC's documentation has good resources on how to effectively use GPU resources on CHTC.

* [Use GPUS](https://chtc.cs.wisc.edu/uw-research-computing/gpu-jobs)
* [Explore and Test Docker Containers](https://chtc.cs.wisc.edu/uw-research-computing/docker-test.html)
* [Use an Apptainer Container in HTC Jobs](https://chtc.cs.wisc.edu/uw-research-computing/apptainer-htc#use-an-apptainer-container-in-htc-jobs)
