#!/bin/bash

docker run --rm -it -d --shm-size=64gb --gpus all \
    -v /home/matykina_ov/RCTrans:/home/docker_rctrans/RCTrans \
    -v /media/matykina_ov/HPR:/home/docker_rctrans/HPR3 \
    --name matykina_rctrans_hpr3  rctrans:latest "/bin/bash"

# docker run --rm -it -d --shm-size=64gb --gpus all \
#     -v /home/matykina_ov/RCTrans:/home/docker_rctrans/RCTrans \
#     -v /media/matykina_ov/HPR:/home/docker_rctrans/HPR3 \
#     --name matykina_rctrans_hpr3 \
#     --user root \
#     rctrans:latest \
#     bash -c 'ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.535.183.01 /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 && tail -f /dev/null'

