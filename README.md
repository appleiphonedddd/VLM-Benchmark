<div align="center">

# 🧠 VLM-Benchmark

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

---

## 🛠️ Environment Setup

Choose **Option A (Conda)** or **Option B (Docker)** depending on your workflow.

### Option A: Conda

```bash
conda env create -f env.yaml
conda activate vlm
```

### Option B: Docker

The environment is pinned for Linux x86_64. GPU evaluation requires an NVIDIA driver and NVIDIA Container Toolkit on the host.

```bash
# Build the Docker image
docker build --platform linux/amd64 -t vlm-benchmark .

# Verify installation
docker run --rm vlm-benchmark --help
```

> **Note:** The Dockerfile uses Miniforge 26.5.3-0 and verifies the installer against its [release checksum](https://github.com/conda-forge/miniforge/releases/tag/26.5.3-0). Local datasets, checkpoints, and results are excluded by `.dockerignore`; mount them into the container when running.

---

## 📐 Run Evaluation

### CLI Usage

```bash
python eval.py \
  --model <model_name> \
  --model_path <hf_path_or_local_path> \
  --benchmark <benchmark_name> \
  --baseline <baseline_name> \
  --batch_size <batch_size>
```

---

### Option A: Running with Conda

```bash
python eval.py \
  --model qwen_vl \
  --model_path Qwen/Qwen3-VL-2B-Instruct \
  --benchmark MMMUPro \
  --baseline fastv \
  --batch_size 1
```

---

### Option B: Running with Docker

Persist results and downloaded models/datasets using volume mounts:

```bash
mkdir -p results

docker run --rm --gpus all \
  -v "$PWD/results:/workspace/results" \
  -v vlm-hf-cache:/root/.cache/huggingface \
  vlm-benchmark \
  --model qwen_vl \
  --model_path Qwen/Qwen3-VL-2B-Instruct \
  --benchmark MMMUPro \
  --baseline fastv \
  --batch_size 1
```
