# Inference

This folder contains scripts for running **model inference and post-hoc analysis** on the ambiguity-aware benchmark.
It supports batch inference, LLM-as-a-judge evaluation, and quantitative/qualitative analysis of model outputs.

The inference pipeline is designed to work with checkpoints trained via **SFT** and **GRPO**, and evaluates whether models respond strategically under visual ambiguity.

---

## Folder Overview

```
inference/
├── acc.py               # Accuracy and score aggregation utilities
├── anal.py              # Main analysis script for inference outputs
├── llm_judge.py         # LLM-as-a-judge evaluation (strategy & factuality)
├── qwen_inf.py          # Qwen-VL inference wrapper
├── run_analysis.sh      # Run analysis pipeline
├── run_llm_judge.sh     # Run LLM-as-a-judge evaluation
├── run_multiple.sh      # Batch inference over multiple checkpoints
```

---

## File Descriptions

### `qwen_inf.py`

Runs inference using Qwen-VL–based models.

* Loads pretrained or fine-tuned checkpoints
* Processes image–question pairs
* Generates model responses for evaluation

---

### `llm_judge.py`

Implements the **LLM-as-a-judge** framework used in the paper.

* Evaluates whether the model’s response strategy is appropriate given ambiguity
* Detects factual distortion
* Produces structured judgment outputs used for GRPO-style scoring

---

### `anal.py`

Performs post-hoc analysis on inference results.

* Aggregates predictions
* Categorizes failure modes (e.g., over-clarification, under-specification)
* Produces statistics used in the Analysis section of the paper

---

### `acc.py`

Utility functions for computing:

* Accuracy
* Strategy correctness
* Aggregate scores across ambiguity types

---

### Shell Scripts

#### `run_analysis.sh`

Runs the full analysis pipeline on saved inference outputs.

```bash
bash run_analysis.sh
```

---

#### `run_llm_judge.sh`

Runs LLM-as-a-judge evaluation on model responses.

```bash
bash run_llm_judge.sh
```

---

#### `run_multiple.sh`

Runs inference across **multiple choice problem** sequentially.

```bash
bash run_multiple.sh
```

---

## Checkpoints & Dataset

Pretrained checkpoints and evaluation datasets can be downloaded from:

**Google Drive**
[https://drive.google.com/drive/folders/1MR5qf5C61TOuv7VOnRXF2DoDAUhgedEe?usp=sharing](https://drive.google.com/drive/folders/1MR5qf5C61TOuv7VOnRXF2DoDAUhgedEe?usp=sharing)

This includes:

* Fine-tuned SFT checkpoints
* GRPO-trained checkpoints
* Ambiguity-aware evaluation datasets

After downloading, update the checkpoint and dataset paths in the scripts as needed.

---

## Notes

* Inference outputs are expected to be saved in JSON/JSONL format for downstream analysis.
* LLM-as-a-judge evaluation assumes access to an API-compatible LLM (e.g., GPT-style models).
* This folder is **read-only with respect to training**; all optimization happens in the `sft/` and `grpo/` directories.

