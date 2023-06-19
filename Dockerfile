FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

ENV DEBIAN_FRONTEND noninteractive

RUN sed -i 's/# deb-src/deb-src/' /etc/apt/sources.list \
    && apt-get update \
    && apt-get build-dep -y python3 \
    && apt-get install -y software-properties-common make git wget curl gnupg \
        apt-transport-https ca-certificates gnupg-agent \
        libsqlite3-dev libssl-dev zlib1g-dev unzip \
        build-essential gdb lcov pkg-config \
        libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev \
        libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev \
        lzma lzma-dev tk-dev uuid-dev zlib1g-dev

COPY . /aprnn_pldi23
WORKDIR /aprnn_pldi23

run make venv -j8

RUN touch /root/.bashrc \
    && echo "source /aprnn_pldi23/external/python_venv/3.9.7/bin/activate" > /root/.bashrc
