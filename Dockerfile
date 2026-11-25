# torch's pip wheels bundle their own CUDA and cuDNN, and the NVIDIA container
# runtime injects the host driver, so a plain Ubuntu base is all that is needed
ARG BASE_IMAGE=docker.io/ubuntu:22.04
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.6

FROM ${UV_IMAGE} AS uv


# ---- Build stage: resolve and install the venv -----------------------------

FROM ${BASE_IMAGE} AS build

COPY --from=uv /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The venv's interpreter symlinks into UV_PYTHON_INSTALL_DIR, so the runtime
# stage must copy it to the same path
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_LINK_MODE=copy

WORKDIR /chatter

# Shallow-clone chatterbox (deep clone is too slow under QEMU emulation)
ARG CHATTERBOX_BRANCH=faster
RUN git clone --depth 1 --branch ${CHATTERBOX_BRANCH} \
    https://github.com/rsxdalv/chatterbox.git /tmp/chatterbox

COPY pyproject.toml uv.lock ./

RUN uv sync --locked --no-editable --no-cache \
      --no-install-package tts-webui-chatterbox-tts && \
    uv pip install --no-cache --no-deps /tmp/chatterbox


# ---- Runtime stage ---------------------------------------------------------

FROM ${BASE_IMAGE}

# ffmpeg transcodes .mp3 voice sources; gcc and libc6-dev are needed by
# torch.compile when --backend inductor is selected
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /chatter

COPY --from=build /opt/python /opt/python
COPY --from=build /chatter/.venv /chatter/.venv
COPY fatterbox ./fatterbox

# Model weights are downloaded here on first start; mount a volume to keep them
VOLUME /chatter/models

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["/chatter/.venv/bin/python", "-m", "fatterbox"]
