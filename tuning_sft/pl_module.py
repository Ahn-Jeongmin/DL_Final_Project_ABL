import json, string, os, re
from typing import Optional, Union
import numpy as np

import torch
# from torch import nn
from torch import distributed as dist
from torch.optim import AdamW, Muon
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import lightning.pytorch as L

from transformers import (
    GenerationConfig,
    Qwen3VLConfig,
    Qwen3VLProcessor,
    # AutoModelForVision2Seq,
    Qwen3VLForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from transformers.generation.utils import GenerateOutput
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast
)
from peft import (
    LoraConfig,
    PeftModelForCausalLM,
    LoraModel,
    TaskType,
    get_peft_model,
)
from utils.metrics import normalize_answer, reform_for_gold, reform_for_pred
import evaluate
from rouge_score import rouge_scorer

from arg_parser import MyNamespace



class MyLightningModule(L.LightningModule):
    def __init__(self, args: MyNamespace) -> None:
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)
        self.train_epoch_losses: list[float] = []
        self.valid_epoch_results: list[np.ndarray] = []
        self.metric_squad = evaluate.load('squad_v2')
        self.metric_rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        self.model = None
        self.processor: Qwen3VLProcessor = None
        
    def configure_model(self) -> None:
        if self.processor is None:
            self.processor = Qwen3VLProcessor.from_pretrained(pretrained_model_name_or_path=self.args.model_config.pretrained_model_name_or_path)
        
        if self.model is not None:
            return
        
        if self.args.valid.gen_config == 'greedy':
            self.generationConfig = GenerationConfig(
                # Class that holds a configuration for a generation task. A `generate` call supports the following generation methods
                # for text-decoder, text-to-text, speech-to-text, and vision-to-text models:
                #     - *greedy decoding* if `num_beams=1` and `do_sample=False`
                #     - *multinomial sampling* if `num_beams=1` and `do_sample=True`
                #     - *beam-search decoding* if `num_beams>1` and `do_sample=False`
                #     - *beam-search multinomial sampling* if `num_beams>1` and `do_sample=True`
                #     - *assisted decoding* if `assistant_model` or `prompt_lookup_num_tokens` is passed to `.generate()`
                ## > Parameters that control the length of the output
                max_new_tokens=100,
                # min_new_tokens=,
                # early_stopping=False,
                ## > Parameters that control the generation strategy used
                do_sample=False,
                # num_beams=1,
                # num_beam_groups=1,
                # penalty_alpha=,
                # use_cache=True,
                ## > Parameters for manipulation of the model output logits
                # temperature=1.0,
                # top_k=50,
                # top_p=1.0,
                # typical_p=1.0,
                # epsilon_cutoff=0.0,
                # eta_cutoff=0.0,
                # diversity_penalty=0.0,
                # repetition_penalty=1.0,
                # encoder_repetition_penalty=1.0,
                # length_penalty=1.0,
                # no_repeat_ngram_size=0,
                # bad_words_ids=,
                # force_words_ids=,
                # renormalize_logits=False,  # 설명에는 True 권장
                # constraints=,
                # forced_bos_token_id=,
                # forced_eos_token_id=,
                # remove_invalid_values=,
                # exponential_decay_length_penalty=,
                # suppress_tokens=,
                # begin_suppress_tokens=,
                # forced_decoder_ids=,
                # sequence_bias=,
                # guidance_scale=,
                # low_memory=
                ## > Parameters that define the output variables of `generate`
                # num_return_sequences=1,
                # output_attentions=False,
                # output_hidden_states=False,
                # output_scores=False,
                # return_dict_in_generate=True,
                ## > Special tokens that can be used at generation time
                # pad_token_id=pad_token_id,
                # bos_token_id=,
                # eos_token_id=eos_token_id,
                ## > Generation parameters exclusive to encoder-decoder models
                ## > Wild card
                # generation_kwargs=,
            )
        elif self.args.valid.gen_config == 'modeldefault':
            # `generation_config` default values have been modified to match model-specific defaults:
            self.generationConfig = GenerationConfig(
                do_sample=True,
                temperature=0.7,
                top_k=20,
                top_p=0.8,
                pad_token_id=151643,
                bos_token_id=151643,
                eos_token_id=[151645, 151643],

                max_new_tokens=100,
            )
            # If this is not desired, please set these values explicitly.
        else:
            raise ValueError(f"Unknown valid.gen_config: {self.args.valid.gen_config}")

        quantization_config = None
        # quantization_config=BitsAndBytesConfig(
        #     # load_in_8bit=False,  # ?
        #     load_in_4bit=True,  # ?
        #     llm_int8_threshold=6.0,  # ?
        #     # llm_int8_skip_modules=None,
        #     # llm_int8_enable_fp32_cpu_offload=False,
        #     # llm_int8_has_fp16_weight=False,
        #     bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16, float16, float32
        #     bnb_4bit_quant_type="nf4",  # fp4, nf4
        #     bnb_4bit_use_double_quant=True,
        # )

        # config = Qwen3VLConfig.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
        vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=self.args.model_config.pretrained_model_name_or_path,
            # config=config,
            torch_dtype=torch.bfloat16,
            device_map=int(os.environ["LOCAL_RANK"]),
            quantization_config=quantization_config,
            # attn_implementation="eager",
        )
        for p in vlm.parameters():
            p.requires_grad = False
        
        vlm.generation_config = self.generationConfig
        # pad_token_id=self.tokenizer.convert_tokens_to_ids('<|endoftext|>')
        # eos_token_id=self.tokenizer.convert_tokens_to_ids(['<|endoftext|>', '<|im_end|>'])

        # peft_config = LoraConfig(
        #     task_type=TaskType.CAUSAL_LM,
        #     inference_mode=False,
        #     # r=self.args.lora_r,
        #     r=8,
        #     target_modules=target_modules,
        #     # lora_alpha=self.args.lora_a,
        #     lora_alpha=8,
        #     lora_dropout=0.1,
        #     # fan_in_fan_out=False,
        #     # bias="none",  # ?
        #     # modules_to_save=['embed_mems'],
        #     # init_lora_weights=True,  # ?
        #     # layers_to_transform=None,
        #     # layers_pattern=None,
        # )
        lora_config = LoraConfig(
            r = self.args.model_config.lora_r,
            lora_alpha = self.args.model_config.lora_a,
            lora_dropout = 0.05,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias = "none",
            task_type = "CAUSAL_LM",
        )
        # self.model: PeftModelForCausalLM|LoraModel|CausalLM = get_peft_model(model=lm, peft_config=peft_config)
        self.model = get_peft_model(model=vlm, peft_config=lora_config)
        self.model.train()
        # self.model.gradient_checkpointing_enable()
        self.model.print_trainable_parameters()  # LoRA 파라미터만 학습되는지 확인
    # # override pl
    # def forward():

    # override pl
    def training_step(self, batch: dict[str, torch.LongTensor], batch_idx) -> dict[str, torch.FloatTensor]:
        batch_inputs: dict[str, torch.LongTensor] = batch['processed']
        outputs: Qwen3VLCausalLMOutputWithPast = self.model.forward(
            **batch_inputs,
        )
        loss: torch.FloatTensor = outputs.loss
        self.train_epoch_losses.append(loss.item())

        self.log(name="train_loss", value=loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_inputs['input_ids'].size(0))
        return {"loss": loss}

    # override pl
    def on_train_epoch_end(self) -> None:
        avg_train_loss: float = np.mean(self.train_epoch_losses)
        self.log(name="avg_train_loss", value=avg_train_loss, on_epoch=True, sync_dist=True)
        self.train_epoch_losses = []

    # override pl
    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx) -> None:  # TODO
        batch_inputs: dict[str, torch.Tensor] = batch['processed']
        batch_question: list[str] = batch['question']
        batch_answer: list[str] = batch['answer']
        batch_uid: list[str] = batch['uid']
        # batch_conversation: list[list[dict]] = batch['conversation']

        outputs: GenerateOutput = self.model.generate(
            **batch_inputs,
            # generation_config=self.generationConfig,
        )

        gened_texts = self.processor.batch_decode(
            # sequences=outputs.sequences[:, batch_inputs['input_ids'].shape[1]:],
            sequences=outputs[:, batch_inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        for uid, gened_text, answer, question in zip(
            batch_uid,
            gened_texts,
            batch_answer,
            batch_question,
            ):

            self.valid_epoch_results.append({
                'uid': uid,
                'output': gened_text,
                'answer': answer,
                'question': question,
            })

    def _validation_epoch_end_metric(self, results: list[dict], split: str='valid'):  # TODO
        dist.barrier()
        object_list = [None] * self.trainer.world_size
        dist.all_gather_object(object_list=object_list, obj=results)
        results = [persample for obj in object_list for persample in obj]
        # print(len(results))
        visit = set()
        uniq_results = []
        for res in results:
            uid = res['uid']
            if uid not in visit:
                visit.add(uid)
                uniq_results.append(res)
        # print(len(uniq_results))
        results_all = sorted(uniq_results, key=lambda x: x['uid'])

        _key_pred_output = 'output'

        results_ambig = [persample for persample in results_all if '_ambig_' in persample['uid']]
        results_unambig = [persample for persample in results_all if '_unambig_' in persample['uid']]


        if split=='valid':
            metric_prefix = 'metric'
        elif split=='test':
            metric_prefix = 'metric_test'
        else:
            raise ValueError(f"Unknown split: {split}")

        for type_name, results in [('all', results_all), ('ambig', results_ambig), ('unambig', results_unambig)]:         
            p_g_pair = list(zip(*[(
                reform_for_pred(uid=persample['uid'], pred=normalize_answer(persample[_key_pred_output])),
                reform_for_gold(uid=persample['uid'], golds=[normalize_answer(persample['answer'])])
            ) for persample in results]))

            predictions = list(p_g_pair[0])
            references = list(p_g_pair[1])
            result_metric_squad = self.metric_squad.compute(predictions=predictions, references=references)

            rouge_scores = [
                self.metric_rouge.score(target=ref['answers']['text'][0], prediction=pred['prediction_text'])
                for ref, pred in zip(references, predictions)
            ]

            # Response Type Accuracy (RTA)
            cnt_same_response_type = 0
            for persample in results:
                gold_text = persample['answer'].strip()
                pred_text = persample[_key_pred_output].strip()

                gold_is_question = gold_text.endswith('?')
                pred_is_question = pred_text.endswith('?')

                if pred_is_question == gold_is_question:
                    cnt_same_response_type += 1

            result_metric = {}
            result_metric.update({k: v for k, v in result_metric_squad.items() if k in ['exact', 'f1', 'total']})

            result_metric['exact'] /= 100.0
            result_metric['f1'] /= 100.0
            result_metric['rouge1'] = sum(score['rouge1'].fmeasure for score in rouge_scores) / len(rouge_scores)
            result_metric['rougeL'] = sum(score['rougeL'].fmeasure for score in rouge_scores) / len(rouge_scores)
            result_metric['rta'] = cnt_same_response_type / len(results)

            print(
                f"{metric_prefix}/EM_{type_name}: {result_metric['exact']:.6f} \t" \
                f"{metric_prefix}/F1_{type_name}: {result_metric['f1']:.6f} \t" \
                f"{metric_prefix}/rouge1_{type_name}: {result_metric['rouge1']:.6f} \t" \
                f"{metric_prefix}/rougeL_{type_name}: {result_metric['rougeL']:.6f} \t" \
                f"{metric_prefix}/RTA_{type_name}: {result_metric['rta']:.6f}"
            )

            self.log(f"{metric_prefix}/EM_{type_name}", result_metric['exact'], sync_dist=True)
            self.log(f"{metric_prefix}/F1_{type_name}", result_metric['f1'], sync_dist=True)
            self.log(f"{metric_prefix}/rouge1_{type_name}", result_metric['rouge1'], sync_dist=True)
            self.log(f"{metric_prefix}/rougeL_{type_name}", result_metric['rougeL'], sync_dist=True)
            self.log(f"{metric_prefix}/RTA_{type_name}", result_metric['rta'], sync_dist=True)
            dist.barrier()

            if self.local_rank==0:
                valid_log_dir = os.path.join(self.args.trainer.output_dir, f"{split}")
                if not os.path.exists(valid_log_dir):
                    os.makedirs(valid_log_dir)
                # file_name_1 = f"_epoch={self.current_epoch}-step={self.global_step}-EM={result_metric['exact']:.6f}-F1={result_metric['f1']:.6f}-include_answer={result_metric['include_answer']:.6f}"
                file_name_1 = f"_epoch={self.current_epoch}-step={self.global_step}-type={type_name}-F1={result_metric['f1']:.6f}-rouge1={result_metric['rouge1']:.6f}-rougeL={result_metric['rougeL']:.6f}-RTA={result_metric['rta']:.6f}"
                file_name = file_name_1+'.jsonl'
                with open(f'{os.path.join(valid_log_dir, file_name)}', 'w') as f:
                    for res in results:
                        f.write(json.dumps(res, ensure_ascii=False)+"\n")
            dist.barrier()

    # override pl
    def on_validation_epoch_end(self):
        self._validation_epoch_end_metric(self.valid_epoch_results, split='valid')
        self.valid_epoch_results = []

    # override pl
    def test_step(self, batch: dict[str, torch.Tensor], batch_idx) -> None:  # TODO
        batch_inputs: dict[str, torch.Tensor] = batch['processed']
        batch_question: list[str] = batch['question']
        batch_answer: list[str] = batch['answer']
        batch_uid: list[str] = batch['uid']
        # batch_conversation: list[list[dict]] = batch['conversation']

        outputs: GenerateOutput = self.model.generate(
            **batch_inputs,
            # generation_config=self.generationConfig,
        )

        gened_texts = self.processor.batch_decode(
            # sequences=outputs.sequences[:, batch_inputs['input_ids'].shape[1]:],
            sequences=outputs[:, batch_inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        for uid, gened_text, answer, question in zip(
            batch_uid,
            gened_texts,
            batch_answer,
            batch_question,
            ):

            self.valid_epoch_results.append({
                'uid': uid,
                'output': gened_text,
                'answer': answer,
                'question': question,
            })

    # override pl
    def on_test_epoch_end(self):
        self._validation_epoch_end_metric(self.valid_epoch_results, split='test')
        self.valid_epoch_results = []
    
    # override pl
    def configure_optimizers(self):
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                        if p.requires_grad
                ],
                "weight_decay": self.args.optimize.weight_decay,
                "lr": self.args.optimize.learning_rate,
            },
        ]
        
        optimizer = AdamW(params=optimizer_grouped_parameters)

        t_total = (
            (1000 // (self.args.trainer.train_batch_size * max(1, self.args.trainer.n_gpu)))
            // self.args.optimize.gradient_accumulation_steps
            * float(self.args.optimize.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.args.optimize.warmup_steps,
            num_training_steps=t_total,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
