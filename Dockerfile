FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# ===============================
# Sistema + Python 3.11.8
# ===============================
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    build-essential \
    cmake \
    libcurl4-openssl-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
        python3.11 \
        python3.11-dev \
        python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

# Python 3.11 como padrão
RUN ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Pip para Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# ===============================
# Ferramentas de build Python
# ===============================
RUN pip install --upgrade \
    pip \
    setuptools \
    wheel

# Corrige compatibilidade (pyparsing + Py 3.11)
RUN pip install --no-cache-dir "pyparsing>=3.1.0"

# ===============================
# PyTorch 2.10 + Unsloth
# ===============================
RUN pip install --no-cache-dir \
    torch==2.10.0 \
    torchvision==0.25.0 \
    torchaudio==2.10.0 \
    "unsloth[colab] @ git+https://github.com/unslothai/unsloth.git"

# ===============================
# Dependências GGUF
# ===============================
RUN pip install --no-cache-dir \
    gguf \
    sentencepiece \
    protobuf

# ===============================
# llama.cpp (clone + build)
# ===============================
ENV LLAMA_CPP_PATH=/opt/llama.cpp

RUN git clone https://github.com/ggerganov/llama.cpp ${LLAMA_CPP_PATH} \
    && cd ${LLAMA_CPP_PATH} \
    && make -j$(nproc)

ENV PATH="${LLAMA_CPP_PATH}/build/bin:${PATH}"

# ===============================
# Dependências do projeto
# ===============================
COPY dataset-financing-infos/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ===============================
# Validação
# ===============================
RUN python - <<EOF
import sys, torch
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.is_available())
EOF
