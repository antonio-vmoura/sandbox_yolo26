# Imagem oficial da NVIDIA com CUDA 12.1 e Ubuntu 22.04
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instala dependências do sistema e Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    && rm -rf /var/lib/apt/lists/*

# Instala o PIP
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py

RUN ln -s /usr/bin/python3.11 /usr/bin/python

WORKDIR /workspace

# Instala PyTorch (CUDA 12.1) e Ultralytics (YOLO) + Jupyter para seus notebooks.
# Todas as versões são fixadas nas usadas no estudo (reprodutibilidade):
# - torch/torchvision/torchaudio: kernels e determinismo variam entre versões;
# - ultralytics: o Tuner (HPO), os defaults de treino e o formato dos
#   checkpoints mudam entre versões, e o checkpoint da Fase 3 (hpo_state.json)
#   recusa retomar uma busca sob outra versão;
# - pandas: usado pelos notebooks de análise (não é dependência do ultralytics).
RUN pip install --upgrade pip && \
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu121 && \
    pip install ultralytics==8.4.21 pandas==3.0.1 jupyterlab

# Copia o código para dentro do container
COPY . /workspace

CMD ["/bin/bash"]