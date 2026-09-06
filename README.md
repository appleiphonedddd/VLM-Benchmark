<div align="center">

# 🧠 VLM-Benchmark

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.x-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

---

## ⚡ Quick Start

```bash
conda env create -f env.yaml
conda activate vlm
```

---

## 📐 Run Evaluation

```bash
python eval.py --model qwen_vl --model_path Qwen/Qwen3-VL-2B-Instruct --benchmark MMMUPro --baseline fastv --batch_size 1
```

