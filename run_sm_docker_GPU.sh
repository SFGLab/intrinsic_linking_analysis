#!/bin/bash

if [[ $# -eq 2 ]]; then
    container_name=$2
elif [[ $# -eq 1 ]]; then
    container_name="md-soft"
else
    echo "Usage: $0"
    exit 2
fi

host_project=$(realpath $1)
container_project="/home/model"


docker run --rm --name ${container_name} \
           --runtime nvidia \
           --gpus all \
           --device /dev/nvidia0 \
           --device /dev/nvidiactl \
           --device /dev/nvidia-modeset \
           --device /dev/nvidia-uvm \
           --device /dev/nvidia-uvm-tools \
           -v ${host_project}:${container_project}/ \
            mkadlof/mdsoft:1.0 \
            ./run.py -c ${container_project}/config.ini
