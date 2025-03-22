# Hello world PyTorch example

Train MNIST using PyTorch in a container.

## Preparing the environment

To ensure that the

```Dockerfile
RUN pixi install --locked --environment prod
```

command will succeed the `pixi.lock` needs to know about the existence of the `prod` environment and be current with it, so run

```
pixi install --environment prod
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
docker run --rm -ti --gpus all uw-madison-dsi/hello_pytorch:jammy-cuda-12.6.0 bash
```

```console
root@5789c0254776:/app# python ./src/torch_detect_GPU.py
PyTorch build CUDA version: 12.6
PyTorch build cuDNN version: 90300
PyTorch build NCCL version: (2, 23, 4)

Number of GPUs found on system: 1

Active GPU index: 0
Active GPU name: NVIDIA GeForce RTX 4060 Laptop GPU
root@5789c0254776:/app# python ./src/torch_MNIST.py
...
Train Epoch: 14 [57600/60000 (96%)]	Loss: 0.005053
Train Epoch: 14 [58240/60000 (97%)]	Loss: 0.006250
Train Epoch: 14 [58880/60000 (98%)]	Loss: 0.001895
Train Epoch: 14 [59520/60000 (99%)]	Loss: 0.002306

Test set: Average loss: 0.0264, Accuracy: 9913/10000 (99%)

root@5789c0254776:/app#
```

or use the `pixi` tasks

```
docker run --rm --gpus all uw-madison-dsi/hello_pytorch:jammy-cuda-12.6.0 pixi run train-mnist
```

or the explicit commands

```
docker run --rm --gpus all uw-madison-dsi/hello_pytorch:jammy-cuda-12.6.0 python ./src/torch_MNIST.py
```
