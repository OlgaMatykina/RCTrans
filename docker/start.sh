#!/bin/bash

docker run --rm -it -d --shm-size=64gb --gpus all \
    -v /home/matykina_ov/RCTrans:/home/docker_rctrans/RCTrans \
    -v /media/matykina_ov/HPR:/home/docker_rctrans/HPR3 \
    --name matykina_rctrans_hpr3  rctrans:latest "/bin/bash"
