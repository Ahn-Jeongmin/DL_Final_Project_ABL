import json
import wandb
import os
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union
from unittest.mock import patch
import numpy as np
import re
import base64
from PIL import Image
import io
import time
import datetime
import random
import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    Qwen3VLProcessor,
    Qwen3VLForConditionalGeneration,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoProcessor,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.import_utils import is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad, selective_log_softmax

import types
from copy import deepcopy

# Import InternVL functions
# from internVL_src import *

from peft import (
    LoraConfig,
    get_peft_model,
)
if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset N times.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        repeat_count (`int`):
            Number of times to repeat each index.

    Example:
    ```python
    >>> sampler = RepeatRandomSampler(["a", "b", "c", "d"], repeat_count=2)
    >>> list(sampler)
    [2, 2, 0, 0, 3, 3, 1, 1]
    ```
    """

    def __init__(self, data_source: Sized, repeat_count: int):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)

    def __iter__(self):
        indexes = [idx for idx in torch.randperm(self.num_samples).tolist() for _ in range(self.repeat_count)]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count


class QwenInternVLGRPOTrainer(Trainer):
    """
    Trainer for GRPO method supporting both Qwen and InternVL models.
    Simplified version that uses LLM-based reward evaluation without grounding tools.
    """

    _tag_names = ["trl", "grpo", "qwen-internvl"]

    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            args: GRPOConfig = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None,
                                                                                                               None),
            peft_config: Optional["PeftConfig"] = None,
            max_pixels: Optional[int] = 256 * 28 * 28,
            min_pixels: Optional[int] = 3136,
            attn_implementation: str = "sdpa",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        args.force_image_size = 448
        args.down_sample_ratio = 0.5
        args.conv_style = "internvl2_5"

        # Model initialization
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation

        # Force bf16 for FlashAttention compatibility
        if args.bf16:
            model_init_kwargs["torch_dtype"] = torch.bfloat16
        elif args.fp16:
            model_init_kwargs["torch_dtype"] = torch.float16
        else:
            # FlashAttention requires fp16 or bf16
            model_init_kwargs["torch_dtype"] = torch.bfloat16
            print("Warning: FlashAttention requires fp16 or bf16. Setting torch_dtype to bfloat16.")

        assert isinstance(model, str), "Model must be a string path for this implementation"
        model_id = model
        self.model_id = model_id

        # Disable caching if gradient checkpointing is enabled
        # model_init_kwargs["use_cache"] = (
        #     False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        # )

        # Load model based on type
        if "qwen" in model_id.lower():
            vlm = Qwen3VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            # checkpoint load
            FT_WEIGHTS = "checkpoint/pytorch_model.bin"  
            lora_config = LoraConfig(
                r = 8,
                lora_alpha = 16,
                lora_dropout = 0.05,
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias = "none",
                task_type = "CAUSAL_LM",
            )
            model = get_peft_model(model=vlm, peft_config=lora_config)
            state_dict = torch.load(FT_WEIGHTS, map_location="cpu")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print("missing keys:", len(missing))
            print("unexpected keys:", len(unexpected))
            model.config.use_cache = True
            model.config.gradient_checkpointing = False

                   
        elif "internvl" in model_id.lower():
            config = InternVLChatConfig.from_pretrained(model_id)
            if config.llm_config.model_type == 'internlm2':
                config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
                print('Using flash_attention_2 for InternLM')
            else:
                config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
                print('Using flash_attention_2 for LLaMA')
            model = InternVLChatModel.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, config=config)
            patch_size = model.config.vision_config.patch_size
            if model.config.vision_config.image_size != args.force_image_size:
                print(f'Resizing position embedding from '
                      f'{model.config.vision_config.image_size} '
                      f'to {args.force_image_size}...')
                model.vision_model.resize_pos_embeddings(old_size=model.config.vision_config.image_size,
                                                         new_size=args.force_image_size,
                                                         patch_size=patch_size)
                model.config.vision_config.image_size = args.force_image_size
            model.config.force_image_size = args.force_image_size
            model.num_image_token = int((args.force_image_size // patch_size) ** 2 * (args.down_sample_ratio ** 2))
        else:
            raise ValueError(f"Unsupported model type: {model_id}")

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "qwen" in model_id.lower():
                self.ref_model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            
            
            elif "internvl" in model_id.lower():
                self.ref_model = InternVLChatModel.from_pretrained(
                    model_id, torch_dtype=torch.bfloat16, config=config)
                if self.ref_model.config.vision_config.image_size != args.force_image_size:
                    print(f'Resizing position embedding from '
                          f'{self.ref_model.config.vision_config.image_size} '
                          f'to {args.force_image_size}...')
                    self.ref_model.vision_model.resize_pos_embeddings(
                        old_size=self.ref_model.config.vision_config.image_size,
                        new_size=args.force_image_size,
                        patch_size=patch_size)
                    self.ref_model.config.vision_config.image_size = args.force_image_size
                self.ref_model.config.force_image_size = args.force_image_size
                self.ref_model.num_image_token = int(
                    (args.force_image_size // patch_size) ** 2 * (args.down_sample_ratio ** 2))
        elif peft_config is None:
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

        # Processing class setup
        if processing_class is None:
            if "qwen" in model_id.lower():
                try:
                    print('qwen_processing')
                    processing_class = AutoProcessor.from_pretrained('Qwen/Qwen3-VL-2B-Instruct')
                except:
                    processing_class = AutoProcessor.from_pretrained('Qwen/Qwen3-VL-2B-Instruct')

                processing_class.pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                processing_class.image_processor.max_pixels = max_pixels
                processing_class.image_processor.min_pixels = min_pixels
            elif "internvl" in model_id.lower():
                # Load pretrained model, tokenizer, and image processor
                tokenizer_path = model_id
                print(f'Loading Tokenizer: {tokenizer_path}')
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=False)
                tokenizer.tokenizer_path = tokenizer_path
                tokenizer.model_max_length = args.max_prompt_length
                token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                              QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                              REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
                num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
                img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
                processing_class = tokenizer
                if num_new_tokens > 0:
                    model.language_model.resize_token_embeddings(len(tokenizer))
                    if self.ref_model is not None:
                        self.ref_model.language_model.resize_token_embeddings(len(tokenizer))
                    output_embeddings = model.language_model.get_output_embeddings().weight.data
                    output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                    output_embeddings[-num_new_tokens:] = output_embeddings_avg
                    model.config.llm_config.vocab_size = len(tokenizer)
                    model.language_model.config.vocab_size = len(tokenizer)
                    if self.ref_model is not None:
                        self.ref_model.config.llm_config.vocab_size = len(tokenizer)
                        self.ref_model.language_model.config.vocab_size = len(tokenizer)
                model.img_context_token_id = img_context_token_id
                model.language_model.config.use_cache = False
                model.vision_model.gradient_checkpointing = False
                model.vision_model.encoder.gradient_checkpointing = False
                if self.ref_model is not None:
                    self.ref_model.img_context_token_id = img_context_token_id
                    self.ref_model.language_model.config.use_cache = False
                    self.ref_model.vision_model.gradient_checkpointing = False
                    self.ref_model.vision_model.encoder.gradient_checkpointing = False

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing classes
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("Number of reward processing classes must match number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.use_vllm = False  # Not using vLLM
        self.beta = args.beta

        # Suppress token estimation warning
        model.warnings_issued["estimate_tokens"] = True

        # Initialize metrics
        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Validation checks for batch size and generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"Global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by number of generations ({self.num_generations}). Valid values: {possible_values}."
            )

        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"Global eval batch size must be evenly divisible by number of generations. "
                    f"Valid values: {possible_values}."
                )

        # Generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            pad_token_id=processing_class.pad_token_id if hasattr(processing_class, 'pad_token_id')
            else processing_class.tokenizer.pad_token_id,
        )

        # Force loss scaling
        self.model_accepts_loss_kwargs = False

        # Add model tags
        self.model.add_model_tags(self._tag_names)

        # Prepare reference model
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        # Prepare reward models
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        # Token IDs
        if "qwen" in model_id.lower():
            self.eos_token_id = processing_class.tokenizer.eos_token_id
            self.pad_token_id = processing_class.tokenizer.pad_token_id
        elif "internvl" in model_id.lower():
            self.eos_token_id = processing_class.convert_tokens_to_ids(EOS_TOKEN)
            self.pad_token_id = processing_class.convert_tokens_to_ids(END_OF_TEXT_TOKEN)

        self.is_train = True

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["message"]

    # def _get_train_sampler(self) -> Sampler:
    #     return RepeatRandomSampler(self.train_dataset, self.num_generations)

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        return RepeatRandomSampler(eval_dataset, 2)  # Faster eval with fewer generations

    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw, logits_to_keep):
        """Get per-token log probabilities for completions."""
        if "qwen" in self.model_id.lower():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        elif "internvl" in self.model_id.lower():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                            pixel_values=pixel_values, image_flags=image_grid_thw)

        logits = outputs.logits
        logits = logits[:, -logits_to_keep - 1:-1, :]  # Exclude last logit
        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)

    def multi_modal_get_item(self, data_items):
        """Process inputs for InternVL model."""
        pixel_values = []
        processed_conversations = []
        num_patches = []
        images = []

        for data_item in data_items:
            # Extract image path
            image_path = [data_item['message'][i]['content'][k]['image']
                          for i in range(len(data_item['message']))
                          for k in range(len(data_item['message'][i]['content']))
                          if 'image' in data_item['message'][i]['content'][k]]
            num_image = len(image_path)
            assert len(image_path) == 1, f'The number of image input should be 1, but got {len(image_path)}.'

            # Always use load_image function to ensure consistent processing
            if isinstance(image_path[0], str):
                pixel_value, image = load_image(image_path[0], self.is_train)
            else:
                # Convert PIL Image to temporary file and process
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    image_path[0].save(tmp_file.name, 'JPEG')
                    pixel_value, image = load_image(tmp_file.name, self.is_train)
                    os.unlink(tmp_file.name)  # Clean up temp file

            images.append(image)
            pixel_values.append(pixel_value)
            num_patches.append(pixel_value.size(0))

            # Extract text safely
            user_text = ""
            for content_item in data_item['message'][0]['content']:
                if content_item.get('type') == 'text':
                    user_text = content_item['text']
                    break
                elif 'text' in content_item:
                    user_text = content_item['text']
                    break
                elif isinstance(content_item, str):
                    user_text = content_item
                    break

            if not user_text:
                # Fallback: try to get any text content
                for content_item in data_item['message'][0]['content']:
                    if isinstance(content_item, dict):
                        for key, value in content_item.items():
                            if key not in ['image', 'type'] and isinstance(value, str):
                                user_text = value
                                break
                    if user_text:
                        break

            # Preprocess conversations
            conversations = [
                {'from': 'human', 'value': f'<image>\n{user_text}'},
            ]

            template_name = self.args.conv_style
            num_image_token_list = [self.model.num_image_token * num_patches[-1]]

            if conversations[0]['from'] == 'system':
                system_prompt = conversations[0]['value']
                conversations = conversations[1:]
            else:
                conv = get_conv_template(template_name)
                system_prompt = conv.system_message

            new_conversations = []
            for conversation in conversations:
                if conversation['from'] == 'human':
                    image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[0]}{IMG_END_TOKEN}'
                    conversation['value'] = conversation['value'].replace('<image>', image_tokens, 1)
                new_conversations.append(conversation)
            conversations = new_conversations

            batches, roles = [], []
            if system_prompt is not None:
                batches.append(f'<|im_start|>system\n{system_prompt}<|im_end|>\n')
                roles.append('system')
            for conversation in conversations:
                if conversation['from'] == 'human':
                    batches.append(f'<|im_start|>user\n{conversation["value"]}<|im_end|>\n')
                    roles.append('human')
                elif conversation['from'] == 'gpt':
                    batches.append(f'<|im_start|>assistant\n{conversation["value"]}<|im_end|>\n')
                    roles.append('gpt')
                else:
                    raise NotImplementedError
            processed_conversations.append(''.join(batches))

        chosen_ret = self.processing_class(
            processed_conversations,
            padding=True,
            padding_side='left',
            return_tensors="pt",
            max_length=self.processing_class.model_max_length,
            truncation=True,
        )

        ret = dict(
            input_ids=chosen_ret['input_ids'],
            attention_mask=chosen_ret['attention_mask'],
            pixel_values=torch.stack([sect for sects in pixel_values for sect in sects], dim=0),
            image_grid_thw=torch.tensor([1 for n in num_patches for i in range(n)], dtype=torch.long),
        )
        return ret, images

    def _prepare_inputs(self, inputs: list, is_train=True) -> dict[str, Union[torch.Tensor, Any]]:
        """Prepare inputs for training/evaluation - supports both Qwen and InternVL."""
        device = self.accelerator.device
        self.is_train = is_train

        # inputs is a list of conversations
        conversations = inputs

        data_types = []
        actual_conversations = []

        for conversation in conversations:
            if isinstance(conversation[-1], dict) and "type" in conversation[-1]:
                data_types.append(conversation[-1]["type"])
                actual_conversations.append(conversation[:-1])
            else:
                data_types.append("Unknown")
                actual_conversations.append(conversation)

        conversations = actual_conversations

        # Initialize completion tracking
        total_completion_mask = torch.zeros(len(conversations), self.max_completion_length, device=device)
        total_completion_ids = torch.full(
            (len(conversations), self.max_completion_length), self.pad_token_id, device=device
        )

        # Convert conversations to the format expected by the processor
        prompts_for_generation = []
        original_prompts = []

        for conversation in conversations:
            # Find the user message
            user_message = None
            for message in conversation:
                if message["role"] == "user":
                    user_message = message
                    break

            if user_message is None:
                raise ValueError("No user message found in conversation")

            # Store original prompt for reward computation
            original_prompts.append(conversation)

            # Create prompt for generation (only user message)
            prompts_for_generation.append([user_message])

        # Process based on model type
        if "qwen" in self.model_id.lower():
            text = [self.processing_class.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            ) for prompt in prompts_for_generation]        
            
            image_inputs, video_inputs = [], []
            for prompt in prompts_for_generation:
                images_in_prompt = []
                videos_in_prompt = []

                for message in prompt:
                    if message["role"] == "user":
                        for content_item in message["content"]:
                            if content_item["type"] == "image":
                                images_in_prompt.append(content_item["image"])
                            elif content_item["type"] == "video":
                                videos_in_prompt.append(content_item["video"])
                
                
                if images_in_prompt:
                    image_inputs.append(images_in_prompt)
                if videos_in_prompt:
                    video_inputs.append(videos_in_prompt)

            # # Clean up text
            for i in range(len(text)):
                if len(text[i].split('<|im_end|>\n<|im_start|>assistant\n')) > 2:
                    t_list = text[i].split('<|im_end|>\n<|im_start|>assistant\n')[:-1]
                    text[i] = '<|im_end|>\n<|im_start|>assistant\n'.join(t_list[:2]) + ''.join(t_list[2:])
            #print(text)
            # Prepare image/video lists
            video_list = [vi for vid_inps in video_inputs for vi in vid_inps] if video_inputs else None
            image_list = [ii for img_inps in image_inputs for ii in img_inps] if image_inputs else None
            # has_images = any(len(imgs) > 0 for imgs in image_inputs)
            # has_videos = any(len(vids) > 0 for vids in video_inputs)
            # Tokenize
            prompt_inputs = self.processing_class(
                text=text,
                images=image_list,
                videos=video_list,
                padding=True,
                padding_side='left',
                return_tensors="pt"
            )

        elif "internvl" in self.model_id.lower():
            # InternVL processing
            # Convert conversations to the format expected by multi_modal_get_item
            data_items = []
            for conversation in conversations:
                data_item = {'message': conversation}
                data_items.append(data_item)

            prompt_inputs, image_inputs = self.multi_modal_get_item(data_items)

        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask, pixel_values = \
            prompt_inputs["input_ids"], prompt_inputs["attention_mask"], prompt_inputs.get("pixel_values")
        image_grid_thw = prompt_inputs.get("image_grid_thw")

        # Ensure vision tensors match model dtype
        if pixel_values is not None:
            model_dtype = next(self.model.parameters()).dtype
            if pixel_values.dtype != model_dtype:
                prompt_inputs["pixel_values"] = pixel_values.to(model_dtype)
                pixel_values = prompt_inputs["pixel_values"]

        # Check prompt length
        if prompt_ids.size(1) > self.max_prompt_length:
            print(f"Warning: prompt length {prompt_ids.size(1)} exceeds max {self.max_prompt_length}")

        # Store original for later use
        original_prompt_ids = prompt_ids.clone()
        original_prompt_mask = prompt_mask.clone()

        start_time = time.time()

        # Generate completions
        with torch.no_grad():
            self.model.eval()
            if "qwen" in self.model_id.lower():
                if self.is_train:
                    
                    self.model.eval()
                    # print(prompt_inputs)
                    # print(prompt_inputs) 
                    # print(prompt_inputs['input_ids'].shape) 
                    # print(prompt_inputs['attention_mask'].shape)
                    prompt_completion_ids = self.model.generate(
                        **prompt_inputs,
                        generation_config=self.generation_config,
                        tokenizer=self.processing_class.tokenizer
                    )
                else:
                    eval_generation_config = GenerationConfig(
                        max_new_tokens=self.max_completion_length,
                        temperature=0.001,
                        do_sample=True,
                        top_k=1,
                        top_p=0.0,
                        pad_token_id=self.processing_class.tokenizer.pad_token_id,
                    )
                    prompt_completion_ids = self.model.generate(
                        **prompt_inputs,
                        tokenizer=self.processing_class.tokenizer,
                        generation_config=eval_generation_config,
                    )

                # Extract completions
                prompt_length = prompt_ids.size(1)
                if not (prompt_ids == prompt_completion_ids[:, :prompt_length]).all():
                    print("Warning: Prompt mismatch detected")
                completion_ids = prompt_completion_ids[:, prompt_length:]

            elif "internvl" in self.model_id.lower():
                if self.is_train:
                    prompt_completion_ids = self.model.generate(
                        input_ids=prompt_inputs["input_ids"],
                        attention_mask=prompt_inputs["attention_mask"],
                        pixel_values=prompt_inputs.get("pixel_values"),
                        generation_config=self.generation_config,
                        tokenizer=self.processing_class
                    )
                else:
                    eval_generation_config = GenerationConfig(
                        max_new_tokens=self.max_completion_length,
                        temperature=0.001,
                        do_sample=True,
                        top_k=1,
                        top_p=0.0,
                        pad_token_id=self.pad_token_id,
                    )
                    prompt_completion_ids = self.model.generate(
                        input_ids=prompt_inputs["input_ids"],
                        attention_mask=prompt_inputs["attention_mask"],
                        pixel_values=prompt_inputs.get("pixel_values"),
                        generation_config=eval_generation_config,
                        tokenizer=self.processing_class
                    )

                completion_ids = prompt_completion_ids

            elif "internvl" in self.model_id.lower():
                if self.is_train:
                    prompt_completion_ids = self.model.generate(
                        input_ids=prompt_inputs["input_ids"],
                        attention_mask=prompt_inputs["attention_mask"],
                        pixel_values=prompt_inputs.get("pixel_values"),
                        generation_config=eval_generation_config,
                        tokenizer=self.processing_class
                    )

                completion_ids = prompt_completion_ids

        time_taken = time.time() - start_time
        # print(f"Generation time: {time_taken:.2f} seconds")

        # Create completion mask (mask everything after first EOS)
        is_eos = (completion_ids == self.eos_token_id) | (completion_ids == self.pad_token_id)
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Copy to total completion tracking
        for i in range(len(completion_ids)):
            actual_length = min(self.max_completion_length, eos_idx[i])
            total_completion_ids[i, :actual_length] = completion_ids[i, :actual_length]
            total_completion_mask[i, :actual_length] = completion_mask[i, :actual_length]

        # Add EOS tokens manually if needed
        for i in range(len(completion_ids)):
            if eos_idx[i] < self.max_completion_length:
                total_completion_ids[i, eos_idx[i]] = self.eos_token_id
                total_completion_mask[i, eos_idx[i]] = 1

        completion_ids = total_completion_ids
        completion_mask = total_completion_mask

        # Prepare for logit computation
        attention_mask = torch.cat([original_prompt_mask, completion_mask], dim=1)
        prompt_completion_ids = torch.cat([original_prompt_ids, completion_ids], dim=1)
        logits_to_keep = completion_ids.size(1)

        start_time = time.time()

        # Compute reference logprobs if training
        if is_train:
            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, attention_mask,
                        pixel_values, image_grid_thw, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, prompt_completion_ids, attention_mask,
                            pixel_values, image_grid_thw, logits_to_keep
                        )
        else:
            ref_per_token_logps = None

        time_taken = time.time() - start_time
        # print(f"Reference logprob time: {time_taken:.2f} seconds")

        start_time = time.time()

        # Decode completions
        if "qwen" in self.model_id.lower():
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        elif "internvl" in self.model_id.lower():
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Convert to the format expected by reward function
        completions = [[{"role": "assistant", "content": completion}] for completion in completions_text]

        # Compute rewards
        rewards_per_func = torch.zeros(len(original_prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
        ):
            reward_start_time = time.time()
            if isinstance(reward_func, nn.Module):
                # For model-based rewards
                messages = [{"messages": p + c} for p, c in zip(original_prompts, completions)]
                texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                # Custom reward function
                simple_images = []
                simple_questions = []

                for conv, data_type in zip(original_prompts, data_types):
                    img = None
                    quest = ""
                    for msg in conv:
                        if msg["role"] == "user":
                            for content in msg["content"]:
                                if content["type"] == "image":
                                    img = content["image"]
                                elif content["type"] == "text":
                                    quest = content["text"]
                    simple_images.append(img)
                    simple_questions.append(quest)

                output_reward_func = reward_func(
                    completions,
                    kwargs={
                        'images': simple_images,
                        'questions': simple_questions,
                        'data_types': data_types
                    }
                )
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

            reward_time = time.time() - reward_start_time
            # print(f"Reward function {i} time: {reward_time:.2f} seconds")

        current_batch_rewards = rewards_per_func.clone()
        rewards_per_func = gather(rewards_per_func)

        # Sum rewards across functions
        rewards = rewards_per_func.sum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        # rewards = rewards.view(-1, self.num_generations)
        # mean_grouped_rewards = rewards.mean(dim=1)
        # std_grouped_rewards = rewards.std(dim=1)


        # Handle case where std is 0 or NaN (single sample per group)
        std_grouped_rewards = torch.where(
            torch.isnan(std_grouped_rewards) | (std_grouped_rewards == 0),
            torch.ones_like(std_grouped_rewards),
            std_grouped_rewards
        )

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # If we have single samples, create artificial advantages
        if advantages.numel() == 1 or torch.all(advantages == 0):
            advantages = torch.randn_like(advantages) * 0.1  # Small random advantages

        # Slice for local process
        process_slice = slice(
            self.accelerator.process_index * len(original_prompts),
            (self.accelerator.process_index + 1) * len(original_prompts),
        )
        advantages = advantages[process_slice]

        # Log metrics
        reward_per_func = current_batch_rewards.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            dataset_name = "default"
            self._metrics[f"{dataset_name}/rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics[f"{dataset_name}/reward"].append(rewards.mean().item())
        self._metrics[f"{dataset_name}/reward_std"].append(std_grouped_rewards.mean().item())

        time_taken = time.time() - start_time
        # print(f"Reward computation time: {time_taken:.2f} seconds")

        # Handle image_inputs for InternVL
        if "internvl" in self.model_id.lower():
            # Convert PIL images to paths or keep as images depending on use case
            image_inputs = [[image] for image in image_inputs]
        elif "qwen" in self.model_id.lower():
            # For consistency, wrap in list format
            if not image_inputs:
                image_inputs = [[] for _ in range(len(conversations))]

        return {
            "step": str(self.state.global_step),
            "input_output_text": [{"conversation": orig, "completion": comp}
                                  for orig, comp in zip(conversations, completions_text)],
            "reward_name": [reward_func.__name__ if hasattr(reward_func, '__name__')
                            else reward_func.config._name_or_path.split("/")[-1] for reward_func in self.reward_funcs],
            "reward_list": current_batch_rewards.tolist(),
            "raw_inputs": conversations,
            "image_inputs": image_inputs,
            "prompt_ids": original_prompt_ids,
            "prompt_mask": original_prompt_mask,
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute GRPO loss."""
        if return_outputs:
            raise ValueError("GRPOTrainer does not support returning outputs")

        start_time = time.time()

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        pixel_values, image_grid_thw = inputs['pixel_values'], inputs['image_grid_thw']
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # Create attention mask
        is_eos = (completion_ids == self.eos_token_id) | (completion_ids == self.pad_token_id)
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=self.accelerator.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=self.accelerator.device).expand(is_eos.size(0), -1)
        completion_mask_no_tool = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        attention_mask = torch.cat([prompt_mask, completion_mask_no_tool], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, pixel_values, image_grid_thw,
                                                    logits_to_keep)

        # Compute KL divergence
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        advantages = inputs["advantages"]

        # GRPO formula
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)

        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        time_taken = time.time() - start_time
        # print(f"Loss computation time: {time_taken:.2f} seconds")
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        """Prediction step for evaluation."""
        # Remove duplicates for faster evaluation
        new_inputs = []
        seen_inputs = set()
        for inp in inputs:
            # Create a simple hash of the input - using string representation
            input_key = str(inp)
            if input_key not in seen_inputs:
                seen_inputs.add(input_key)
                new_inputs.append(inp)
        inputs = new_inputs

        inputs = self._prepare_inputs(inputs, is_train=False)

        # Log results to wandb (random sampling for efficiency)
        if self.accelerator.is_main_process and random.random() < 0.3:
            columns = ["image", "question", "prediction", "reward"]
            data = []

            # Get first sample for logging
            if inputs['image_inputs'] and len(inputs['image_inputs']) > 0:
                image_list = inputs['image_inputs'][0]
                if image_list and len(image_list) > 0:
                    # Make sure we're getting the actual PIL Image
                    first_image = image_list[0] if isinstance(image_list[0], Image.Image) else image_list[0][
                        0] if isinstance(image_list[0], list) else None
                    if first_image and hasattr(first_image, 'mode'):  # Check if it's a PIL Image
                        wandb_image = wandb.Image(first_image)
                    else:
                        wandb_image = None
                else:
                    wandb_image = None
            else:
                wandb_image = None

            # Extract question from conversation
            conversation = inputs['raw_inputs'][0]
            question = ""
            for message in conversation:
                if message["role"] == "user":
                    for content in message["content"]:
                        if content["type"] == "text":
                            question = content["text"]
                            break
                    break

            prediction = inputs['input_output_text'][0]['completion']
            reward = inputs['reward_list'][0]

            data.append([wandb_image, question, prediction, reward])

            table = wandb.Table(columns=columns, data=data)
            step = inputs['step']
            wandb.log({f"eval_step_{step}": table})

        # Save local logs
        local_log_path = os.path.join(
            self.args.output_dir,
            f"local_log_step_{inputs['step']}",
            f"evaluation_results_{self.accelerator.process_index}.json"
        )
        os.makedirs(os.path.dirname(local_log_path), exist_ok=True)

        existing_logs = []
        if os.path.exists(local_log_path):
            with open(local_log_path, 'r') as f:
                existing_logs = json.load(f)

        for i in range(len(inputs['input_output_text'])):
            # Extract question from conversation
            conversation = inputs['raw_inputs'][i]
            question = ""
            for message in conversation:
                if message["role"] == "user":
                    for content in message["content"]:
                        if content["type"] == "text":
                            question = content["text"]
                            break
                    break

            existing_logs.append({
                "question": question,
                "prediction": inputs['input_output_text'][i]['completion'],
                "reward_names": inputs['reward_name'],
                "rewards": inputs['reward_list'][i],
            })

        with open(local_log_path, 'w') as f:
            json.dump(existing_logs, f, indent=2)

        # Compute loss if training
        if self.is_train:
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inputs)
                loss = loss.mean().detach()
            return loss, None, None
        else:
            return torch.tensor(0.0, device=self.accelerator.device), None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """Log metrics with proper averaging."""
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}

        # Add eval_ prefix if this is evaluation logging
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()
