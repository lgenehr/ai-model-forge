FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Sistema
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    cmake \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Python padrão
RUN ln -s /usr/bin/python3 /usr/bin/python

# Atualiza pip
RUN pip install --upgrade pip

# Copia requirements do projeto
COPY dataset-financing-infos/requirements.txt /tmp/requirements-dataset.txt
COPY dataset-financing-infos/finetune/requirements.txt /tmp/requirements-finetune.txt

RUN pip install --no-cache-dir \
    -r /tmp/requirements-dataset.txt \
    -r /tmp/requirements-finetune.txt