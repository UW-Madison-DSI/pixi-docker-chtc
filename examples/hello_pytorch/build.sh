#!/bin/bash

docker system prune -f

docker pull ghcr.io/prefix-dev/pixi:jammy
docker pull ghcr.io/prefix-dev/pixi:jammy-cuda-11.8.0

docker build \
    --file Dockerfile \
    --platform linux/amd64 \
    --tag uw-madison-dsi/hello_pytorch:jammy-cuda-11.8.0 \
    .

docker images uw-madison-dsi/hello_pytorch
