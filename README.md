# ClarifyVLM: A Vision-Language Model that Asks Back to Resolve Visual Ambiguity

### Contribution
[![GitHub](https://img.shields.io/badge/GitHub-Bae--Minwook-181717?logo=github&logoColor=white)](https://github.com/minwook09)
[![GitHub](https://img.shields.io/badge/GitHub-Ahn--Jeongmin-181717?logo=github&logoColor=white)](https://github.com/Ahn-Jeongmin)
[![GitHub](https://img.shields.io/badge/GitHub-Lee--Gunhwi-181717?logo=github&logoColor=white)](https://github.com/gkzmsltm)


### Original Repository
[DL_Final_Project_ABL.git](https://github.com/minwook09/DL_Final_Project_ABL.git)

---
This repository contains the full training and evaluation pipeline for **ambiguity-aware Vision–Language Models (VLMs)**.
The project investigates whether models can *strategically recognize and resolve visual ambiguity*, rather than blindly producing overconfident answers.

The pipeline consists of:
1. Ambiguity-aware dataset generation
2. Supervised Fine-Tuning (SFT)
3. Reinforcement Learning with Group Relative Policy Optimization (GRPO)
4. Inference and analysis with LLM-as-a-judge evaluation

---

## Environment Setup (uv)

This project uses **uv** for fast and reproducible Python environment management.

### 1. Create Virtual Environment

```bash
uv venv
source .venv/bin/activate
````

### 2. Install Dependencies

```bash
uv pip install -r requirements.txt
```

Python version is specified in `.python-version` and dependency versions are locked via `uv.lock`.

> ⚠️ CUDA-enabled PyTorch is required for training and inference.

---

## Repository Structure

```
.
├── dataset_generation/     # Ambiguity-aware dataset construction
├── tuning_sft/             # Supervised fine-tuning (SFT)
├── tuning_grpo/            # GRPO reinforcement learning
├── inference/              # Inference, analysis, and LLM-as-a-judge
│
├── main.py                 # High-level entry (optional orchestration)
├── run.sh                  # Example end-to-end execution script
│
├── requirements.txt
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Pipeline Overview

### Dataset Generation (`dataset_generation/`)

Constructs ambiguity-aware image–question pairs.

* Explicitly controls ambiguity types
* Produces structured annotations for strategy-aware learning

See `dataset_generation/README.md` for details.

---

### Supervised Fine-Tuning (`tuning_sft/`)

Trains the base VLM to imitate reference responses.

* Learns basic answering and clarification behaviors
* Does **not** explicitly optimize strategic correctness

See `tuning_sft/README.md`.

---

### GRPO Training (`tuning_grpo/`)

Optimizes strategic behavior under ambiguity using **Group Relative Policy Optimization**.

* Uses LLM-as-a-judge–based reward functions
* Compares multiple generations per input
* Penalizes overconfidence and factual distortion

See `tuning_grpo/README.md`.

---

### Inference & Analysis (`inference/`)

Evaluates trained models.

* Batch inference
* LLM-as-a-judge evaluation
* Failure mode and bias analysis

See `inference/README.md`.

---

## Datasets & Checkpoints

All preprocessed datasets and pretrained checkpoints are available at:

📂 **Google Drive**
[https://drive.google.com/drive/folders/1MR5qf5C61TOuv7VOnRXF2DoDAUhgedEe?usp=sharing](https://drive.google.com/drive/folders/1MR5qf5C61TOuv7VOnRXF2DoDAUhgedEe?usp=sharing)

This includes:

* Dataset generation outputs
* SFT checkpoints
* GRPO checkpoints

---

## Notes

* SFT checkpoints are required before GRPO training.
* GRPO training is sensitive to reward stability and generation group size.
* Inference assumes JSON/JSONL formatted outputs.

