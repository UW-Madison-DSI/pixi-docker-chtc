#!/bin/bash

docker system prune -f

docker pull tonistiigi/binfmt
docker pull ghcr.io/prefix-dev/pixi:noble
docker pull ghcr.io/prefix-dev/pixi:noble-cuda-12.9

docker build \
    --file Dockerfile \
    --platform linux/amd64 \
    --tag ghcr.io/uw-madison-dsi/pixi-docker-chtc:hello-pytorch-noble-cuda-12.9 \
    .

docker images ghcr.io/uw-madison-dsi/pixi-docker-chtc
