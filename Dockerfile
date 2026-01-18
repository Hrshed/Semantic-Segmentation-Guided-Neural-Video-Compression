FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS main

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

ENV UV_SYSTEM_PYTHON=1

COPY pyproject.toml ./

RUN uv sync
RUN uv pip install jupyterlab
RUN uv pip install wandb

COPY ./ext/simple-waymo-open-dataset-reader ./ext/simple-waymo-open-dataset-reader
RUN cd ./ext/simple-waymo-open-dataset-reader && uv run setup.py install


# COPY ./checkpoints ./checkpoints

COPY ./src ./src
COPY ./trainer_video_model.py ./
COPY ./video_compression_config.yaml ./