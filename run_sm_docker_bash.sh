#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Usage: $0"
    exit 2
fi

host_project=$(realpath $1)
container_project="/home/model/"

docker run -it --rm --name md-soft \
           --runtime nvidia \
           --gpus all \
           --device /dev/nvidia0 \
           --device /dev/nvidiactl \
           --device /dev/nvidia-modeset \
           --device /dev/nvidia-uvm \
           --device /dev/nvidia-uvm-tools \
           -v ${host_project}:${container_project} \
            mkadlof/mdsoft:1.0 \
            /bin/bash
