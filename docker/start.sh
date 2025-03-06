#!/bin/bash

docker run --rm -it -d --shm-size=64gb --gpus '"device=1"' \
    -v /home/matykina_ov/RCTrans:/home/docker_rctrans/RCTrans \
    -v /datasets:/home/docker_rctrans/HPR1 \
    --name matykina_rctrans  rctrans:latest "/bin/bash"
