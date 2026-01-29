FROM unslothai/unsloth:latest

# Install system dependencies required for llama.cpp conversion and other tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements and install dependencies
# We copy to a temporary location to ensure installation succeeds during build
# regardless of how volumes are mounted later.
COPY dataset-financing-infos/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
