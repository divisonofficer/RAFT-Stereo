# Use nvidia/cuda base image with Ubuntu 20.04 and CUDA 11.3
#FROM nvidia/cuda:11.8.1-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    sudo \
    python3 \
    python3-pip \
    python3-dev \
    p7zip-full \
    libgl1-mesa-glx libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install pytorch==2.3.0 torchvision==0.18.0 pytorch-cuda=11.8 \
    matplotlib tensorboard scipy opencv-python tqdm opt-einsum \
    imageio scikit-image ipywidgets

# Clone RAFT-Stereo repository
RUN git clone https://github.com/princeton-vl/RAFT-Stereo.git /RAFT-Stereo

# Set working directory
WORKDIR /RAFT-Stereo

# Download pretrained models
RUN bash download_models.sh

# Perform extra setup in sampler directory
RUN cd sampler && python3 setup.py install && cd ..

# Set the default command to start bash
CMD ["bash"]