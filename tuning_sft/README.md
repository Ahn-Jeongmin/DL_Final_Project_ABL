# Supervised Fine-Tuning (tuning_sft)

This folder contains code for **Supervised Fine-Tuning (SFT)** of Vision–Language Models on the ambiguity-aware dataset.
SFT serves as the **initial training stage** before reinforcement learning with GRPO.

The goal of this stage is to teach the model basic response patterns—answering, asking clarification questions, and following instruction formats—without explicit optimization of strategic behavior.

---

## Directory Structure

```
tuning_sft/
├── utils/                 # Helper modules
├── arg_parser.py          # Argument definitions
├── datamodules.py         # Dataset loading and preprocessing
├── main_train.py          # SFT training entry point
├── pl_module.py           # PyTorch Lightning training module
├── run_0.sh               # SFT run (config variant 0)
├── run_1.sh               # SFT run (config variant 1)
├── run_2.sh               # SFT run (config variant 2)
```

---

## File Descriptions

### `main_train.py`

Main entry point for supervised fine-tuning.

* Parses training arguments
* Initializes dataset and model
* Launches PyTorch Lightning training loop

```bash
python main_train.py --config CONFIG
```

---

### `pl_module.py`

PyTorch Lightning module.

* Defines forward pass and loss computation
* Handles training and validation steps
* Manages logging and checkpointing

---

### `datamodules.py`

Dataset and dataloader definitions.

* Loads ambiguity-aware image–question pairs
* Handles train/validation splits
* Applies preprocessing and batching logic

---

### `arg_parser.py`

Centralized argument management.

* Training hyperparameters
* Dataset paths
* Model and optimizer settings

---

### `utils/`

Utility functions used across training scripts.

* Logging helpers
* Data formatting utilities
* Miscellaneous shared functions

---

## Training Scripts

### `run_0.sh`, `run_1.sh`, `run_2.sh`

These scripts run the **same training pipeline** with **different argument configurations** (e.g., dataset variants, hyperparameters, or seeds).

```bash
bash run_0.sh
bash run_1.sh
bash run_2.sh
```

Typical differences include:

* Dataset split or ambiguity subset
* Learning rate or batch size
* Random seed

This setup allows controlled comparison across SFT configurations.

---

## Dataset & Checkpoints

Preprocessed datasets and pretrained checkpoints can be downloaded from:

**Google Drive**
[https://drive.google.com/drive/folders/1MR5qf5C61TOuv7VOnRXF2DoDAUhgedEe?usp=sharing](https://drive.google.com/drive/folders/1MR5qf5C61TOuv7VOnRXF2DoDAUhgedEe?usp=sharing)

Place datasets and checkpoints according to the paths specified in the run scripts.

---

## Notes

* SFT optimizes for imitation of reference responses, not strategic correctness.
* Models trained with SFT alone may over-ask or under-clarify under ambiguity.
* SFT checkpoints are intended to be used as initialization for GRPO training.

