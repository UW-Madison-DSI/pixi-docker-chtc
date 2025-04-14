#!/bin/bash

docker system prune -f

docker pull tonistiigi/binfmt
docker pull ghcr.io/prefix-dev/pixi:noble
docker pull ghcr.io/prefix-dev/pixi:noble-cuda-12.6.3

docker build \
    --file Dockerfile \
    --platform linux/amd64 \
    --tag uw-madison-dsi/hello_pytorch:noble-cuda-12.6.3 \
    .

docker images uw-madison-dsi/hello_pytorch
