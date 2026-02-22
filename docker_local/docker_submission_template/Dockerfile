FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Miniconda (Python 3.10)
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -O /tmp/miniconda.sh \
 && /bin/bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
 && rm /tmp/miniconda.sh
ENV PATH=${CONDA_DIR}/bin:$PATH

SHELL ["/bin/bash", "-lc"]
RUN conda create -y -n app python=3.10 \
 && conda clean -ya
ENV PATH=/opt/conda/envs/app/bin:$PATH

# Install Python deps
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 -r /tmp/requirements.txt \
 && rm -rf /tmp/pip-unpack-* /tmp/pip-install-* /tmp/*

# Copy code
WORKDIR /workspace
# copy entire submission template context into workspace (self-contained build context)
COPY . /workspace

ENV PYTHONPATH=/workspace

ENTRYPOINT ["python", "/workspace/run_infer.py"]
