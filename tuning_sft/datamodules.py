import pathlib
import json
import jsonlines
from PIL import Image
import torch
from torch.utils.data import Dataset
import lightning as L
from transformers import Qwen3VLProcessor
from arg_parser import MyNamespace


class VQADataset(Dataset):  # TODO
    def __init__(
        self,
        args: MyNamespace,
        # tokenizer: LlamaTokenizer,
        split: str,
        # task: str,
        num_samples: int=None,
    ) -> None:
        self.args = args
        self.split = split

        PATH_DATASET = "dataset"

        PATH_IMAGE_TRAIN_AMBIG = pathlib.Path(PATH_DATASET, "train_ambig_raw_images")
        PATH_IMAGE_TRAIN_UNAMBIG = pathlib.Path(PATH_DATASET, "train_unambig_raw_images")
        PATH_IMAGE_TEST_AMBIG = pathlib.Path(PATH_DATASET, "test_ambig_raw_images")
        PATH_IMAGE_TEST_UNAMBIG = pathlib.Path(PATH_DATASET, "test_unambig_raw_images")

        PATH_JSONL_TRAIN_AMBIG = pathlib.Path(PATH_DATASET, "train_ambig_qa_description.jsonl")
        PATH_JSONL_TRAIN_UNAMBIG = pathlib.Path(PATH_DATASET, "train_unambig_qa_description.jsonl")
        PATH_JSONL_TEST_AMBIG = pathlib.Path(PATH_DATASET, "test_ambig_qa_description.jsonl")
        PATH_JSONL_TEST_UNAMBIG = pathlib.Path(PATH_DATASET, "test_unambig_qa_description.jsonl")

        idx = 0
        if split == "train":
            idx_2 = 0
            with jsonlines.open(PATH_JSONL_TRAIN_AMBIG, "r") as f:
                idx_3 = 0
                ambig = []
                for sample in f:
                    qa = json.loads(sample["qa"])
                    if idx % 10 != 0:
                        ambig.append({
                            'image_filename': sample["image_filename"],
                            'image_path': pathlib.Path(PATH_IMAGE_TRAIN_AMBIG, sample["image_filename"]),
                            'entity_description': sample["entity_description"],
                            'question': qa["question"],
                            'answer': qa["answers"]["4"],
                            'uid': f"{'train'}_{idx:04d}_{split}_{idx_2:04d}_{'ambig'}_{idx_3:04d}",
                        })
                        idx_2 += 1
                        idx_3 += 1
                    idx += 1
            with jsonlines.open(PATH_JSONL_TRAIN_UNAMBIG, "r") as f:
                idx_3 = 0
                unambig = []
                for sample in f:
                    qa = json.loads(sample["qa"])
                    if idx % 10 != 0:
                        unambig.append({
                            'image_filename': sample["image_filename"],
                            'image_path': pathlib.Path(PATH_IMAGE_TRAIN_UNAMBIG, sample["image_filename"]),
                            'entity_description': sample["entity_description"],
                            'question': qa["question"],
                            'answer': qa["answers"]["1"],
                            'uid': f"{'train'}_{idx:04d}_{split}_{idx_2:04d}_{'unambig'}_{idx_3:04d}",
                        })
                        idx_2 += 1
                        idx_3 += 1
                    idx += 1
        elif split == "valid":
            idx_2 = 0
            with jsonlines.open(PATH_JSONL_TRAIN_AMBIG, "r") as f:
                idx_3 = 0
                ambig = []
                for sample in f:
                    qa = json.loads(sample["qa"])
                    if idx % 10 == 0:
                        ambig.append({
                            'image_filename': sample["image_filename"],
                            'image_path': pathlib.Path(PATH_IMAGE_TRAIN_AMBIG, sample["image_filename"]),
                            'entity_description': sample["entity_description"],
                            'question': qa["question"],
                            'answer': qa["answers"]["4"],
                            'uid': f"{'train'}_{idx:04d}_{split}_{idx_2:04d}_{'ambig'}_{idx_3:04d}",
                        })
                        idx_2 += 1
                        idx_3 += 1
                    idx += 1
            with jsonlines.open(PATH_JSONL_TRAIN_UNAMBIG, "r") as f:
                idx_3 = 0
                unambig = []
                for sample in f:
                    qa = json.loads(sample["qa"])
                    if idx % 10 == 0:
                        unambig.append({
                            'image_filename': sample["image_filename"],
                            'image_path': pathlib.Path(PATH_IMAGE_TRAIN_UNAMBIG, sample["image_filename"]),
                            'entity_description': sample["entity_description"],
                            'question': qa["question"],
                            'answer': qa["answers"]["1"],
                            'uid': f"{'train'}_{idx:04d}_{split}_{idx_2:04d}_{'unambig'}_{idx_3:04d}",
                        })
                        idx_2 += 1
                        idx_3 += 1
                    idx += 1
        elif split == "test":
            with jsonlines.open(PATH_JSONL_TEST_AMBIG, "r") as f:
                idx_2 = 0
                ambig = []
                for sample in f:
                    qa = json.loads(sample["qa"])
                    ambig.append({
                        'image_filename': sample["image_filename"],
                        'image_path': pathlib.Path(PATH_IMAGE_TEST_AMBIG, sample["image_filename"]),
                        'entity_description': sample["entity_description"],
                        'question': qa["question"],
                        'answer': qa["answers"]["4"],
                        'uid': f"{split}_{idx:04d}_{'ambig'}_{idx_2:04d}",
                    })
                    idx_2 += 1
                    idx += 1
            with jsonlines.open(PATH_JSONL_TEST_UNAMBIG, "r") as f:
                idx_2 = 0
                unambig = []
                for sample in f:
                    qa = json.loads(sample["qa"])
                    unambig.append({
                        'image_filename': sample["image_filename"],
                        'image_path': pathlib.Path(PATH_IMAGE_TEST_UNAMBIG, sample["image_filename"]),
                        'entity_description': sample["entity_description"],
                        'question': qa["question"],
                        'answer': qa["answers"]["1"],
                        'uid': f"{split}_{idx:04d}_{'unambig'}_{idx_2:04d}",
                    })
                    idx_2 += 1
                    idx += 1
        else:
            raise NotImplementedError

        self.dataset = ambig+unambig

        if num_samples is not None and num_samples > 0:
            print(num_samples)
            self.dataset = self.dataset.select(range(num_samples))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        inputs = self.dataset[index]

        image = Image.open(inputs['image_path']).convert("RGB")
        question = inputs["question"]
        answer = inputs["answer"]
        uid = inputs["uid"]


        conversation = []

        if self.args.dataset.system_prompt_version == 0:
            sys_prompt = None
        elif self.args.dataset.system_prompt_version == 1:
            # ahn: llm as judge 활용할 때 생성형은 이거 활용했습니다. 4지선다에서 마지막 문장 뺀 프롬프트에요
            sys_prompt = "You are an AI language model that provides answers based on the given image and text input.\n" \
                            "Analyze the image and text carefully to generate accurate and relevant responses."
        elif self.args.dataset.system_prompt_version == 2:
            # lee: chatgpt
            sys_prompt = '''You are a vision-language assistant specialized in answering questions about images.

Your behavior must follow these rules:

1. Base all answers strictly on the visual information in the provided image.
2. When the question is clear and the referenced object is unambiguous, answer directly and concisely.
3. When the question references an object that:
   - does not appear in the image, OR
   - appears multiple times without a clear way to distinguish which instance is intended, OR
   - could refer to more than one plausible entity in the image,
   you must NOT guess or assume.

4. In such ambiguous situations, DO NOT answer the question.
   Instead, ask the user a clarification question to identify which object they are referring to.
   Examples:
   - “Which man do you mean, the one on the left or the one holding the bag?”
   - “There are several cups in the image. Which one are you asking about?”
   - “Do you mean the larger dog or the smaller dog?”

5. If the question requires information not visually present (e.g., identity, name, unseen attributes), say you cannot determine that from the image.

6. Never fabricate details or hallucinate objects.

Follow these rules consistently for every interaction.'''
        else:
            # ahn: 저 4지선다 시스템프롬프트는 이거 사용했고
            sys_prompt = "You are an AI language model that provides answers based on the given image and text input.\n" \
                            "Analyze the image and text carefully to generate accurate and relevant responses.\n" \
                            "Your answer should be only a number (1, 2, 3, or 4) corresponding to the most appropriate answer choice."

            raise NotImplementedError

        if sys_prompt is not None:
            conversation.append({
                "role": "system",
                "content": [
                    {"type": "text", "text": sys_prompt}
                ]
            })

        conversation.extend([
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ])

        return {
            "image": image,
            "question": question,
            "answer": answer,
            "conversation": conversation,
            "uid": uid,
        }


class VQADataModule(L.LightningDataModule):
    def __init__(self, args: MyNamespace) -> None:
        super().__init__()
        self.args = args
        # self.processor = processor
        self.processor = Qwen3VLProcessor.from_pretrained(pretrained_model_name_or_path="Qwen/Qwen3-VL-2B-Instruct")
        self.train_batch_size = self.args.trainer.train_batch_size
        self.valid_batch_size = self.args.trainer.eval_batch_size
        self.train_num_workers = self.args.dataset.train_num_workers
        self.valid_num_workers = self.args.dataset.valid_num_workers
        self.num_samples_train = self.args.debug.n_train
        self.num_samples_valid = self.args.debug.n_valid
        
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def setup(self, stage: str) -> None:
        if self.train_dataset is None:
            self.train_dataset = VQADataset(
                args=self.args,
                split="train",
                num_samples=self.num_samples_train,
            )
        if self.valid_dataset is None:
            self.valid_dataset = VQADataset(
                args=self.args,
                split="valid",
                num_samples=self.num_samples_valid,
            )
        if self.test_dataset is None:
            self.test_dataset = VQADataset(
                args=self.args,
                split="test",
                num_samples=self.num_samples_valid,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.train_num_workers,
            collate_fn=self._collate_fn_train,
            pin_memory=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.valid_batch_size,
            shuffle=False,
            num_workers=self.valid_num_workers,
            collate_fn=self._collate_fn_valid,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.valid_batch_size,
            shuffle=False,
            num_workers=self.valid_num_workers,
            collate_fn=self._collate_fn_valid,
            pin_memory=True,
        )

    def _collate_fn_train(self, batch):
        images = [item["image"] for item in batch]
        # questions = [item["question"] for item in batch]
        # answers = [item["answer"] for item in batch]
        # uids = [item["uid"] for item in batch]
        
        conversations = [item["conversation"] for item in batch]
        processed_text = self.processor.apply_chat_template(
            conversation=conversations,
            add_generation_prompt=False,
            tokenize=False,
        )
        processed = self.processor(
            images=images,
            text=processed_text,
            return_tensors="pt",
            padding=True,
            padding_side="right",
        )
        
        conversations_for_label = [item["conversation"][:-1] for item in batch]
        processed_text_for_label = self.processor.apply_chat_template(
            conversation=conversations_for_label,
            add_generation_prompt=True,
            tokenize=False,
        )
        processed_for_label = self.processor(
            images=images,
            text=processed_text_for_label,
            return_tensors="pt",
            padding=True,
            padding_side="right",
        )

        labels = processed["input_ids"].clone()
        labels[processed["attention_mask"] == 0] = -100
        
        len_label_mask = processed_for_label["attention_mask"].size(1)
        labels[:, :len_label_mask][processed_for_label["attention_mask"] == 1] = -100
        processed["labels"] = labels

        return {
            "processed": processed,
        }

    def _collate_fn_valid(self, batch):
        images = [item["image"] for item in batch]
        questions = [item["question"] for item in batch]
        answers = [item["answer"] for item in batch]
        uids = [item["uid"] for item in batch]
        
        conversations = [item["conversation"][:-1] for item in batch]
        processed_text = self.processor.apply_chat_template(
            conversation=conversations,
            add_generation_prompt=True,
            tokenize=False,
        )
        processed = self.processor(
            images=images,
            text=processed_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        )

        return {
            "processed": processed,
            "question": questions,
            "answer": answers,
            "uid": uids,
            "conversation": conversations,
        }
