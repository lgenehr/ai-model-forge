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
    curl \
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
# Ferramentas Python
# ===============================
RUN python -m pip install --upgrade pip setuptools wheel

# ===============================
# Constraints (Python 3.11 fix)
# ===============================
COPY constraints.txt /tmp/constraints.txt

# ===============================
# PyTorch 2.10 + Unsloth
# ===============================
# Torch sem dependências (evita resolution-too-deep)
RUN python -m pip install --no-cache-dir \
    torch==2.10.0 \
    torchvision==0.25.0 \
    torchaudio==2.10.0 \
    --no-deps

# Ecossistema controlado
RUN python -m pip install --no-cache-dir \
    -c /tmp/constraints.txt \
    unsloth @ git+https://github.com/unslothai/unsloth.git \
    bitsandbytes \
    accelerate \
    trl \
    peft

# ===============================
# Dependências GGUF
# ===============================
RUN python -m pip install --no-cache-dir \
    -c /tmp/constraints.txt \
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
# Copia os requirements (como você pediu)
# ===============================
COPY dataset-financing-infos/requirements.txt /tmp/requirements-dataset.txt
COPY dataset-financing-infos/finetune/requirements.txt /tmp/requirements-finetune.txt

# ===============================
# Dependências do projeto
# ===============================
RUN python -m pip install --no-cache-dir \
    -c /tmp/constraints.txt \
    -r /tmp/requirements-dataset.txt \
    -r /tmp/requirements-finetune.txt

# ===============================
# Validação
# ===============================
RUN python - <<EOF
import sys, torch, pyparsing,
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("pyparsing:", pyparsing.__version__)
print("GPU:", torch.cuda.is_available())
EOF
