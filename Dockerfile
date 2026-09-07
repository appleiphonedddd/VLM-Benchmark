FROM nvidia/cuda:13.3.0-cudnn-devel-ubuntu24.04

ARG MINIFORGE_VERSION=26.5.3-0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/conda/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# env.yaml contains Linux x86_64 package builds.
RUN test "$(uname -m)" = x86_64 \
    && cd /tmp \
    && curl -fsSLO "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
    && curl -fsSLO "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh.sha256" \
    && sha256sum -c "Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh.sha256" \
    && bash "Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" -b -p /opt/conda \
    && rm -f Miniforge3-*.sh Miniforge3-*.sh.sha256

WORKDIR /workspace

COPY env.yaml /workspace/env.yaml
RUN conda env create -n vlm -f /workspace/env.yaml \
    && conda clean -ya

ENV PATH="/opt/conda/envs/vlm/bin:$PATH"

COPY . /workspace

ENTRYPOINT ["python", "eval.py"]
CMD ["--help"]
