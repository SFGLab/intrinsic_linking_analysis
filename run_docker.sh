#!/usr/bin/env bash

if [[ $# -ne 2 ]]; then
    echo "Usage: run_docker image_name project_path"
    exit 2
fi

docker_image_name=$1
project_name=knots
container_user_name=uknots

host_project=$(realpath $2)
host_data="${host_project}/data"
host_results="${host_project}/results"

container_project="/home/${container_user_name}/${project_name}/"
container_data="${container_project}/data"
container_results="${container_project}/results"

docker run --rm -it \
       -v "${host_project}":"${container_project}":ro \
       -v "${host_data}":"${container_data}":ro \
       -v "${host_results}":"${container_results}" \
       -w "${container_project}" \
       "${docker_image_name}"
