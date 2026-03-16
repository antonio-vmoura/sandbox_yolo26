# YOLO26 Fine-Tuning

This repository provides scripts and configuration files for fine-tuning the **Ultralytics YOLO26** model in a custom dataset, focusing on skin lesion instance segmentation.

The goal is to evaluate how well YOLO26 adapts to medical images with limited data, leveraging its state-of-the-art segmentation capabilities.

https://www.ultralytics.com/blog/how-to-custom-train-ultralytics-yolo26-for-instance-segmentation

---

## Overview

The training pipeline includes:

* Loading and preparing the dataset (Roboflow YOLO format)
* Fine-tuning YOLO26 for instance segmentation
* Automatic saving of checkpoints, confusion matrices, and metrics
* Fully reproducible execution via Docker with GPU support

---

## Requirements

* Docker with NVIDIA GPU support
* NVIDIA Container Toolkit installed
* Dataset (in YOLO format) available at:

```
./datasets/<dataset_name>

```

---

## Expected Project Structure

```
sandbox_sam3/
│
├── logs/               # Training outputs
├── dataset/            # Dataset
├── yolo26_seg/         # Model source code
└── utils/              # Useful scripts
```

---

## Environment Setup

### Build the Docker Image

Build the environment containing CUDA, PyTorch, and all necessary dependencies:

```bash
docker build -t yolo26_ft .
```

---

## Training Execution

### Option A: Run training using ALL available GPUs

```bash
docker run --gpus all -it --rm \
  --ipc=host \
  --user $(id -u):$(id -g) \
  -e TORCH_HOME=/workspace/cache/torch \
  -e HOME=/workspace/cache \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $(pwd)/datasets:/workspace/datasets \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/yolo26_seg:/workspace/yolo26_seg \
  -v $(pwd)/utils:/workspace/utils \
  -v $(pwd)/cache:/workspace/cache \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  yolo26_ft \
  python /workspace/yolo26_seg/train.py 2>&1 | tee logs/yolo26_ft_ph2.log

```

### Option B: Run training using a SINGLE GPU

```bash
docker run --gpus '"device=0"' -it --rm \
  --ipc=host \
  --user $(id -u):$(id -g) \
  -e TORCH_HOME=/workspace/cache/torch \
  -e HOME=/workspace/cache \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v $(pwd)/datasets:/workspace/datasets \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/yolo26_seg:/workspace/yolo26_seg \
  -v $(pwd)/utils:/workspace/utils \
  -v $(pwd)/cache:/workspace/cache \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  yolo26_ft \
  python /workspace/yolo26_seg/train.py 2>&1 | tee logs/yolo26_ft_ph2_gpu0.log

```

### Option C: Automated Training (Wait for Free GPUs)

```bash
chmod +x wait_gpu.sh
```

```bash
./wait_gpu.sh
```

---

## Running on a Remote Server

### Run training in the background

Create a screen session:

```bash
screen -S yolo26_ft

```

Run the Docker command normally. Detach while keeping the process running:

```text
Ctrl + A, then D

```

Reattach later:

```bash
screen -r yolo26_ft

```

---

### Copy results from the server

```bash
rsync -avz --progress -e "ssh -p 13508 -v" antoniovinicius@164.41.75.221:/home/antoniovinicius/projects/SANDBOX_YOLO26/logs/ph2_finetuning /home/avmoura_linux/Documents/unb/SANDBOX_YOLO26

```

---

### Environment Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install ultralytics jupyterlab
```

---

### Hardware Monitoring

```bash
nvidia-smi
nvtop
```

---