# Hello world PyTorch example

Train MNIST using PyTorch in a container.

## Preparing the environment

To ensure that the

```Dockerfile
RUN pixi install --locked --environment prod
```

command will succeed the `pixi.lock` needs to know about the existence of the `prod` environment and be current with it, so run

```
pixi lock
```

before the build.

## Building the container image locally

```
bash build.sh
```

## Running container locally

> [!TIP]
> For nice GPU use tracking in an `htop` like manner install [`nvtop` ](https://github.com/Syllo/nvtop)
>
> ```console
> $ pixi global install nvtop
> $ nvtop
> ```

```
docker run --rm -ti --gpus all ghcr.io/uw-madison-dsi/hello-pytorch:noble-cuda-12.6.3 bash
```

```console
root@5789c0254776:/app# python ./src/torch_detect_GPU.py
PyTorch build CUDA version: 12.6
PyTorch build cuDNN version: 90800
PyTorch build NCCL version: (2, 26, 2)

Number of GPUs found on system: 1

Active GPU index: 0
Active GPU name: NVIDIA GeForce RTX 4060 Laptop GPU
root@5789c0254776:/app# python ./src/torch_MNIST.py
...
Train Epoch: 14 [57600/60000 (96%)]	Loss: 0.001337
Train Epoch: 14 [58240/60000 (97%)]	Loss: 0.009490
Train Epoch: 14 [58880/60000 (98%)]	Loss: 0.005959
Train Epoch: 14 [59520/60000 (99%)]	Loss: 0.003723

Test set: Average loss: 0.0276, Accuracy: 9916/10000 (99%)

root@5789c0254776:/app#
```

or use the `pixi` tasks

```
docker run --rm --gpus all ghcr.io/uw-madison-dsi/hello-pytorch:noble-cuda-12.6.3 pixi run train-mnist
```

or the explicit commands

```
docker run --rm --gpus all ghcr.io/uw-madison-dsi/hello-pytorch:noble-cuda-12.6.3 python ./src/torch_MNIST.py
```
