#!/bin/bash

# Download the training data locally to transfer to the worker node
if [ ! -f "MNIST_data.tar.gz" ]; then
    # c.f. https://github.com/CHTC/templates-GPUs/blob/450081144c6ae0657123be2a9a357cb432d9d394/shared/pytorch/MNIST_data.tar.gz
    curl -sLO https://raw.githubusercontent.com/CHTC/templates-GPUs/450081144c6ae0657123be2a9a357cb432d9d394/shared/pytorch/MNIST_data.tar.gz
fi

condor_submit -interactive hello_pytorch_gpu.sub
