import os
import torch
import json

from transformers import (
    Qwen3VLConfig,
    Qwen3VLProcessor,
    # AutoModelForVision2Seq,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLTextModel
)
import pathlib
import jsonlines

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel
)
from PIL import Image


class Infeqwen:
    def __init__(self, inf_type):
        BASE_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
        
        if inf_type == 'qwen_base':
            print(inf_type)
            self.model=Qwen3VLForConditionalGeneration.from_pretrained(
                BASE_MODEL_ID,
                dtype=torch.bfloat16,    
            )
            self.model = self.model.to("cuda") 
        elif inf_type == 'sft':
            print(inf_type)
            vlm= Qwen3VLForConditionalGeneration.from_pretrained(
                BASE_MODEL_ID,
                dtype=torch.bfloat16,    
            )

            lora_config = LoraConfig(
                r = 8,
                lora_alpha = 16,
                lora_dropout = 0.05,
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                bias = "none",
                task_type = "CAUSAL_LM",
            )
            FT_WEIGHTS = "checkpoint/sft_checkpoint.bin"
            self.model = get_peft_model(model=vlm, peft_config=lora_config)
            self.model = self.model.to("cuda") 
            state_dict = torch.load(FT_WEIGHTS, map_location="cpu")

            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            # print(missing[0], unexpected[0])
            print("missing keys:", len(missing))
            print("unexpected keys:", len(unexpected))
            
        elif inf_type == 'sft_grpo':
            print(inf_type)
            FT_WEIGHTS = "checkpoint/grpo_checkpoint"

            self.model = PeftModel.from_pretrained(
                Qwen3VLForConditionalGeneration.from_pretrained(
                    BASE_MODEL_ID, torch_dtype=torch.bfloat16), 
                FT_WEIGHTS
            )
            self.model = self.model.to("cuda")
        else:
            raise NotImplemented
        self.processor = Qwen3VLProcessor.from_pretrained(BASE_MODEL_ID)
 
            
    def inference_qwen(self, text, image_path):
        
        image = Image.open(image_path).convert("RGB").resize((224,224))
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text[0])
        return output_text[0]
