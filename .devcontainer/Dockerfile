# syntax=docker/dockerfile:1.7
# Meridian developer container.
# Base: official NVIDIA CUDA 12.6 devel image on Ubuntu 24.04.
# Targets fleet GPUs of June 2026: Ada (sm_89), Hopper (sm_90), Blackwell (sm_100).

FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    CARGO_HOME=/home/vscode/.cargo \
    RUSTUP_HOME=/home/vscode/.rustup \
    PATH=/home/vscode/.cargo/bin:/usr/local/cuda/bin:$PATH

# ---------------------------------------------------------------------------
# OS packages: build essentials, mdbook deps, Python 3.11.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        git-lfs \
        gnupg \
        libssl-dev \
        ninja-build \
        pkg-config \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip \
        sudo \
        unzip \
        zip \
        zsh \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python.
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# ---------------------------------------------------------------------------
# Non-root user.
# ---------------------------------------------------------------------------
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
 && useradd  --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
 && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
 && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
WORKDIR /home/$USERNAME

# ---------------------------------------------------------------------------
# Rust 1.85 via rustup (pinned by rust-toolchain.toml at repo root).
# ---------------------------------------------------------------------------
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain 1.85.0 --profile minimal \
            --component rustfmt clippy rust-src rust-analyzer
RUN cargo install --locked cargo-nextest cargo-deny cargo-cyclonedx mdbook release-plz maturin

# ---------------------------------------------------------------------------
# uv: fast Python package manager.
# ---------------------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH=/home/$USERNAME/.local/bin:$PATH

# ---------------------------------------------------------------------------
# Working directory mounted by VS Code.
# ---------------------------------------------------------------------------
WORKDIR /workspaces/meridian
