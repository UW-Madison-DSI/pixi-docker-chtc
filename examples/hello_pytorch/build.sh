#!/bin/bash

docker build \
    --file Dockerfile \
    --platform linux/amd64 \
    --tag uw-madison-dsi/hello_pytorch:jammy-cuda-11.8.0 \
    .
