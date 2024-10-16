#!/usr/bin/env bash

if [[ $# -ne 1 ]]; then
    echo "Usage: build_docker docker_build_path"
    exit 2
fi

docker_build_path=$1
docker_image_name=$(basename "${docker_build_path}")

docker build . -f "${docker_build_path}"/Dockerfile -t "${docker_image_name}"

# RUN git clone git@bitbucket.org:4dnucleome/md_soft.git
# RUN git clone git@bitbucket.org:mkadlof/structuregenerator.git

# RUN git clone git@bitbucket.org:4dnucleome/md_soft.git && \
#     cd md_soft && \
#     pip install -r requirements.txt

# RUN git clone git@bitbucket.org:mkadlof/structuregenerator.git
