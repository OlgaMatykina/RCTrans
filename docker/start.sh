#!/bin/bash

docker run --rm -it -d --shm-size=64gb --gpus all \
    -v /home/matykina_ov/RCTrans:/home/docker_rctrans/RCTrans \
    -v /media/matykina_ov/HPR1:/home/docker_rctrans/HPR1 \
    -v /media/matykina_ov/data:/home/docker_rctrans/data \
    -v /home/matykina_ov:/home/docker_rctrans/HPR2 \
    --name matykina_rctrans_hpr2  rctrans:latest "/bin/bash"
