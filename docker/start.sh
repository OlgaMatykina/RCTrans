#!/bin/bash

docker run --rm -it -d --shm-size=64gb --gpus all \
    -v /media/matykina_ov/FastSSD/RCTrans:/home/docker_rctrans/RCTrans \
    -v /media/matykina_ov/HPR1:/home/docker_rctrans/HPR1 \
    -v /media/matykina_ov/data:/home/docker_rctrans/data \
    -v /home/matykina_ov:/home/docker_rctrans/HPR2 \
    -v /media/matykina_ov/FastSSD:/home/docker_rctrans/HPR3 \
    --name matykina_rctrans_hpr3  rctrans:latest "/bin/bash"
