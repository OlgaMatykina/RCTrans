#!/bin/bash

docker run --rm -it -d --shm-size=64gb --gpus '"device=0,1"' \
    -v /home/matykina_ov/RCTrans:/home/docker_rctrans/RCTrans \
    -v /datasets/nuScenes2d:/home/docker_rctrans/HPR3 \
    --name matykina_rctrans_hpr3  rctrans:latest "/bin/bash"
