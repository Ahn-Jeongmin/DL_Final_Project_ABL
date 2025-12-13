from accelerate import Accelerator
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor, HfArgumentParser
from datasets import Dataset
from trl.trainer.grpo_config import GRPOConfig
from peft import LoraConfig, get_peft_model
from multiprocessing import Pool
from tqdm import tqdm
from reward import *
from utils import *
from CustomGRPO import QwenInternVLGRPOTrainer
from collections import defaultdict, Counter

os.environ["TOKENIZERS_PARALLELISM"] = "false"
accelerator = Accelerator()

from collections import defaultdict, Counter


def load_data(jsonl_data):
    data_list = []
    with open(jsonl_data, 'r', encoding='utf-8') as file:
        for line in file:
            data_list.append(json.loads(line.strip()))

    return data_list


def process_dataset(data_list, dataset_type):
    print(f"\nProcessing {dataset_type} dataset:")
    print(f"Total samples: {len(data_list)}")

    random.shuffle(data_list)

    dataset = []
    for data in tqdm(data_list, desc=f"Processing {dataset_type} data"):
        dataset.append(process_data(data))

    return dataset


def main():
    train_jsonl = 'dataset/train.jsonl'
    val_jsonl = 'dataset/val.jsonl'

    train_raw = load_data(train_jsonl)
    val_raw = load_data(val_jsonl)

    train_dataset = process_dataset(train_raw, "train")[:10]
    val_dataset = process_dataset(val_raw, "valid")[:4]
    print(f"\nFinal results:")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Valid dataset size: {len(val_dataset)}")

    train_types = Counter(data['level'] for data in train_raw)
    valid_types = Counter(data['level'] for data in val_raw)
    print(f"\nTrain type distribution: {dict(train_types)}")
    print(f"Valid type distribution: {dict(valid_types)}")

    model_name = "SFT_CHECKPOINT_PATH"
    output_dir = "GRPO"
    run_name = "GRPO"

    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=run_name,
        learning_rate=5e-6,
        num_generations=2,
        per_device_eval_batch_size=4,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        torch_empty_cache_steps=1,
        max_completion_length=200,
        max_prompt_length=250,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        num_train_epochs=3,
        report_to="wandb",
        logging_steps=1,
        beta=0.01,
        bf16=True,
        bf16_full_eval=True,
        log_completions=True,
        disable_tqdm=False,
        push_to_hub=False,
        lr_scheduler_type="cosine",
    )

    end_flag = False
    if os.path.exists(training_args.output_dir):
        checkpoint_list = [d for d in os.listdir(training_args.output_dir) if d.endswith('end_of_training.txt')]
        if len(checkpoint_list) > 0:
            print(f"Training has been finished. Please remove {training_args.output_dir} to continue training.")
            end_flag = True

    if not end_flag:
        trainer = accelerator.prepare(QwenInternVLGRPOTrainer(
            model='Qwen/Qwen3-VL-2B-Instruct',
            args=training_args,
            reward_funcs=[llm_judge_reward_function],
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=None,
            reward_processing_classes=None,
            peft_config=None,
        ))

        trainer.train()

        with open(os.path.join(training_args.output_dir, "end_of_training.txt"), "w") as f:
            f.write("Training finished.\n")
            trainer.accelerator.wait_for_everyone()
            accelerator.end_training()
            trainer.accelerator.clear()


if __name__ == '__main__':
    main()
