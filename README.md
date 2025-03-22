# Example configurations for using containerized Pixi environments with HTCondor and GPUs

These examples are inspired by the [Center for High Throughput Computing examples](https://github.com/CHTC/templates-GPUs), the [`pixi-docker` project](https://github.com/prefix-dev/pixi-docker), and [Pavel Zwerschke](https://github.com/pavelzw)'s [`pixi-docker-example`](https://github.com/pavelzw/pixi-docker-example)s.

## CUDA enabled Docker images

These examples assume that you want to use GPU resources to take advantage of hardware acceleration and so focus on using the [Pixi Docker base images](https://github.com/prefix-dev/pixi-docker/pkgs/container/pixi) built on the [NVIDIA CUDA enabled images](https://github.com/NVIDIA/nvidia-docker) for runtime use with the the NVIDIA Container Toolkit.

### Local installation

- Make sure that you have the [`nvidia-container-toolkit`](https://github.com/NVIDIA/nvidia-container-toolkit) installed on the host machine
- Check the [list of available tags on Github Container Registry](https://github.com/prefix-dev/pixi-docker/pkgs/container/pixi) to find the tag you want

> [!TIP]
> You can use [`crane`](https://github.com/google/go-containerregistry/tree/main/cmd/crane) to do this from the command line as well
>
> ```console
> $ pixi global install crane
> $ crane ls ghcr.io/prefix-dev/pixi
> ```

- Use `docker pull` to pull down the image corresponding to the tag

Example:

```
docker pull ghcr.io/prefix-dev/pixi:0.39.0-jammy-cuda-12.6.3
```

### Local use

To check that NVIDIA GPUS are being properly detected run

```
docker run --rm --gpus all ghcr.io/prefix-dev/pixi:0.39.0-jammy-cuda-12.6.3 'nvidia-smi'
```

and check if the [`nvidia-smi`](https://developer.nvidia.com/nvidia-system-management-interface) output appears correctly.

To run (interactively) using GPUs on the host machine:

```
docker run --rm -ti --gpus all ghcr.io/prefix-dev/pixi:0.39.0-jammy-cuda-12.6.3
```
