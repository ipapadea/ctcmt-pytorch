FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git wget curl ca-certificates \
    build-essential ninja-build pkg-config

# Ensure python3 is the default python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system --uid ${USER_ID} ctcmt -g sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER ctcmt
WORKDIR /home/ctcmt

# Upgrade pip, setuptools, wheel
RUN pip install --user --upgrade pip setuptools wheel

# Install PyTorch 2.0+ with CUDA 12.4
RUN pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install core dependencies
RUN pip install --user \
    pyyaml \
    numpy \
    scipy \
    pillow \
    scikit-image \
    opencv-python \
    pycocotools \
    tensorboard \
    tqdm

# Add local pip install to PATH
ENV PATH="/home/ctcmt/.local/bin:${PATH}"

# Set working directory for the application
WORKDIR /home/ctcmt/ctcmt-pytorch-clean

# Default command
CMD ["/bin/bash"]
