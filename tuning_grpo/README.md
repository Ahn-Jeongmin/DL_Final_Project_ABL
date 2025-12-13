# GRPO Training (tuning_grpo)

This folder contains the implementation of **Group Relative Policy Optimization (GRPO)** used to fine-tune Vision–Language Models for **ambiguity-aware strategic behavior**.

Unlike standard supervised fine-tuning (SFT), GRPO explicitly optimizes *how* a model responds under ambiguity—rewarding appropriate clarification, cautious reasoning, and penalizing overconfident or factually distorted answers.

---

## Directory Structure

```
tuning_grpo/
├── dataset/              # Training / validation data for GRPO
├── CustomGRPO.py         # Custom GRPO trainer implementation
├── internVL_src.py       # InternVL / VLM model interface
├── main.py               # Training entry point
├── reward.py             # Reward function definitions
├── utils.py              # Utility functions
├── vision_process.py     # Image preprocessing utilities
├── run_train.sh          # Training script
```

---

## File Descriptions

### `main.py`

Entry point for GRPO training.

* Loads model and dataset
* Initializes GRPO trainer
* Runs training and checkpointing

```bash
python main.py
```

---

### `CustomGRPO.py`

Core implementation of **Group Relative Policy Optimization**.

* Groups multiple generations per input
* Computes relative rewards within each group
* Stabilizes policy updates under ambiguous supervision

This module replaces standard PPO-style optimization with group-relative comparison.

---

### `reward.py`

Defines reward functions under an **LLM-as-a-judge** framework.

* Strategy correctness (answer vs. clarification)
* Factual distortion detection
* Partial penalties for strategically correct but factually flawed outputs

The reward formulation directly reflects the evaluation protocol used in the paper.

---

## Dataset

The `dataset/` directory contains ambiguity-aware training data used for GRPO.
Preprocessed datasets can be downloaded from:

**Google Drive**
[https://drive.google.com/drive/folders/1MR5qf5C61TOuv7VOnRXF2DoDAUhgedEe?usp=sharing](https://drive.google.com/drive/folders/1MR5qf5C61TOuv7VOnRXF2DoDAUhgedEe?usp=sharing)

After downloading, place the dataset under:

```bash
tuning_grpo/dataset/
```

---

## Training

To start GRPO training:

```bash
bash run_train.sh
```

This script:

* Loads a pretrained SFT checkpoint
* Performs GRPO optimization
* Saves intermediate and final checkpoints

---

## Notes

* GRPO is sensitive to batch size and number of generations per input.
* Reward stability depends on consistent LLM-as-a-judge behavior.
* Training logs and checkpoints should be monitored closely to avoid mode collapse.
