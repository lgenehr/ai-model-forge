FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# ===============================
# Sistema
# ===============================
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

# Pip
RUN pip install --upgrade pip

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

# Expor binários no PATH
ENV PATH="${LLAMA_CPP_PATH}/build/bin:${PATH}"

# ===============================
# Dependências do projeto
# (sem torch / unsloth / cuda)
# ===============================
COPY dataset-financing-infos/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ===============================
# Validação básica (opcional)
# ===============================
RUN python - <<EOF
import torch, unsloth
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU disponível:", torch.cuda.is_available())
EOF
