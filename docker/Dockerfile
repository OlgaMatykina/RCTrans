FROM puzlcloud/pytorch:1.10.1-cuda11.3-cudnn8-jupyter-g1-1.1.0-python3.8

# ENV http_proxy=http://matykina:Y1Qe3CYQM1lx@195.26.226.45:3128
# ENV https_proxy=http://matykina:Y1Qe3CYQM1lx@195.26.226.45:3128

# RUN echo 'Acquire::http::Proxy "http://matykina:Y1Qe3CYQM1lx@195.26.226.45:3128";' > /etc/apt/apt.conf.d/95proxies && \
#     echo 'Acquire::https::Proxy "http://matykina:Y1Qe3CYQM1lx@195.26.226.45:3128";' >> /etc/apt/apt.conf.d/95proxies

# add user and his password
ARG USER=docker_rctrans
ARG UID=1053
ARG GID=1053
# default password
ARG PW=user
RUN id

# Instal basic utilities
RUN sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends python3.8-dev git wget unzip bzip2 sudo build-essential ca-certificates && \
    sudo apt-get install ffmpeg libsm6 libxext6  -y && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*

# Create the user
RUN sudo useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | sudo chpasswd && sudo adduser ${USER} sudo
RUN id
USER ${USER}
WORKDIR /home/${USER}
USER ${UID}:${UID}

RUN pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/cu113/torch_stable.html

RUN pip install flash-attn==0.2.2 --no-build-isolation && \
    pip install mmdet==2.28.2 && \
    pip install mmsegmentation==0.30.0

RUN git clone https://github.com/liyih/RCTrans

# Переход в папку репозитория и установка зависимостей
RUN cd ./RCTrans/mmdetection3d && \
    pip install -v -e . && \
    cd ..

RUN pip install ipython && \
    pip install fvcore && \
    pip install spconv-cu111==2.1.21 && \
    pip install yapf==0.40.0 && \
    pip install setuptools==59.5.0 && \
    pip install ccimport==0.3.7 && \
    pip install pccm==0.3.4 && \
    pip install timm && \
    pip3 install wandb && \
    pip install pytorch-lightning==1.6.0

RUN pip install mmcv-full==1.7.0 --force --no-cache-dir -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html && \
    pip install mmengine