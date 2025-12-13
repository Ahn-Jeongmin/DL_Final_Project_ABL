import argparse
from typing import Union


def check_number(value: str) -> Union[int, float]:
    try:
        if '.' in value:
            return float(value)
        else:
            return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid value: {value}")


def add_lightning_trainer_argument(parser: argparse.ArgumentParser):
    # Pytorch Lightning setting - env
    parser.add_argument('--trainer_seed', type=int, default=42, dest='trainer.seed')
    parser.add_argument('--trainer_accelerator', type=str, default="gpu", dest='trainer.accelerator')
    parser.add_argument('--trainer_strategy', type=str, default="ddp", dest='trainer.strategy',
        choices=[
            'ddp',
            'zero_1',
            'zero_1_offload_opt',
            'zero_2',
            'zero_2_offload_opt',
        ]
    )
    parser.add_argument('--trainer_precision', type=str, default="bf16-mixed", choices=['32-true', 'bf16-mixed', '16-mixed'], dest='trainer.precision')
    parser.add_argument('--trainer_n_gpu', type=int, default=1, dest='trainer.n_gpu')
    # Pytorch Lightning setting - fit train
    parser.add_argument('--trainer_train_batch_size', type=int, default=128, dest='trainer.train_batch_size')
    # parser.add_argument('--ckpt_monitor', type=str, default='recall', choices=['recall', 'train_loss'])
    # parser.add_argument('--early_stop_callback', type=int, default=0, choices=[0, 1])  # unused
    # # Pytorch Lightning setting - fit valid
    parser.add_argument('--trainer_eval_batch_size', type=int, default=1, dest='trainer.eval_batch_size')
    # parser.add_argument('--limit_val_batches', type=float, default=1.0)
    # parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--trainer_val_check_interval', type=check_number, default=0.5, dest='trainer.val_check_interval')
    # Pytorch Lightning setting - save load
    parser.add_argument('--trainer_log_root', type=str, default='logs', dest='trainer.log_root')
    parser.add_argument('--trainer_output_dir', type=str, default=None, dest='trainer.output_dir')
    # parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    # parser.add_argument('--certain_epoch', type=int, default=None)
    # parser.add_argument('--given_ckpt', type=str, default='')
    # parser.add_argument('--infer_ckpt', type=str, default='')
    parser.add_argument('--trainer_init_ckpt', type=str, default='', dest='trainer.init_ckpt')
    return parser


class MyNamespace_trainer(argparse.Namespace):
    def __init__(self) -> None:
        super().__init__()
        self.seed: int
        self.accelerator: str
        self.strategy: str
        self.precision: str
        self.n_gpu: int
        self.train_batch_size: int
        self.eval_batch_size: int
        self.val_check_interval: int
        self.log_root: str
        self.output_dir: str
        self.init_ckpt: str


def add_optimize_argument(parser: argparse.ArgumentParser):
    # train setting - optimizer hyperparameters
    parser.add_argument('--optimize_weight_decay', type=float, default=1e-4, dest='optimize.weight_decay')
    parser.add_argument('--optimize_adam_epsilon', type=float, default=1e-8, dest='optimize.adam_epsilon')
    parser.add_argument('--optimize_max_grad_norm', type=float, default=1.0, dest='optimize.max_grad_norm')
    parser.add_argument('--optimize_learning_rate', type=float, default=2e-4, dest='optimize.learning_rate')
    # train setting - scheduler hyperparameters
    parser.add_argument('--optimize_warmup_steps', type=int, default=0, dest='optimize.warmup_steps')
    parser.add_argument('--optimize_num_train_epochs', type=int, default=500, dest='optimize.num_train_epochs')
    parser.add_argument('--optimize_gradient_accumulation_steps', type=int, default=1, dest='optimize.gradient_accumulation_steps')
    return parser


class MyNamespace_optimize(argparse.Namespace):
    def __init__(self) -> None:
        super().__init__()
        self.weight_decay: float
        self.adam_epsilon: float
        self.max_grad_norm: float
        self.learning_rate: float
        self.warmup_steps: int
        self.num_train_epochs: int
        self.gradient_accumulation_steps: int


def add_model_config_argument(parser: argparse.ArgumentParser):
    # model architecture config
    parser.add_argument('--model_config_pretrained_model_name_or_path', type=str, default='Qwen/Qwen3-VL-2B-Instruct', dest='model_config.pretrained_model_name_or_path')
    parser.add_argument('--model_config_lora_r', type=int, default=64, dest='model_config.lora_r')
    parser.add_argument('--model_config_lora_a', type=int, default=16, dest='model_config.lora_a')
    return parser


class MyNamespace_model_config(argparse.Namespace):
    def __init__(self) -> None:
        super().__init__()
        self.pretrained_model_name_or_path: str
        self.lora_r: int
        self.lora_a: int


def add_debug_argument(parser: argparse.ArgumentParser):
    # debug
    parser.add_argument('--debug_test1000', type=int, default=0, help='default to 0,1', choices=[0, 1], dest='debug.test1000')
    parser.add_argument('--debug_n_train', type=int, default=None, dest='debug.n_train')
    parser.add_argument('--debug_n_valid', type=int, default=None, dest='debug.n_valid')
    parser.add_argument('--debug_n_test', type=int, default=None, dest='debug.n_test')
    parser.add_argument('--debug_log_version', type=int, default=None, dest='debug.log_version')
    return parser


class MyNamespace_debug(argparse.Namespace):
    def __init__(self) -> None:
        super().__init__()
        self.test1000: int
        self.n_train: int
        self.n_valid: int
        self.n_test: int
        self.log_version: int


def add_valid_argument(parser: argparse.ArgumentParser):
    # # valid setting - generate option
    parser.add_argument('--valid_gen_config', type=str, default="modeldefault", dest='valid.gen_config', choices=['greedy', 'modeldefault'])
    # parser.add_argument('--length_penalty', type=int, default=0.8)
    # parser.add_argument('--num_beams', type=int, default=5, help='generated id num (include invalid)')
    # # valid setting - metric option
    return parser


class MyNamespace_valid(argparse.Namespace):
    def __init__(self) -> None:
        super().__init__()
        self.gen_config: str


def add_dataset_argument(parser: argparse.ArgumentParser):
    # task & split setting
    # parser.add_argument('--dataset_valid_task', type=str, default='nq_open', dest='dataset.valid_task', choices=['nq_open', 'tqa_open'])
    # parser.add_argument('--dataset_valid_split', type=str, default='valid', dest='dataset.valid_split', choices=['train', 'valid'])
    # train setting
    parser.add_argument('--dataset_system_prompt_version', type=int, default=0, dest='dataset.system_prompt_version')
    parser.add_argument('--dataset_entity_description', type=int, default=0, dest='dataset.entity_description', choices=[0, 1])
    # # in&out setting
    return parser


class MyNamespace_dataset(argparse.Namespace):
    def __init__(self) -> None:
        super().__init__()
        # self.valid_task: str
        # self.valid_split: str
        self.train_num_workers: str
        self.valid_num_workers: str
        self.system_prompt_version: int
        self.entity_description: int


class MyNamespace(argparse.Namespace):
    def __init__(self) -> None:
        super().__init__()
        self.trainer=MyNamespace_trainer()
        self.optimize=MyNamespace_optimize()
        self.model_config=MyNamespace_model_config()
        self.debug=MyNamespace_debug()
        self.valid=MyNamespace_valid()
        self.dataset=MyNamespace_dataset()


def post_process_args(parsed_args: argparse.Namespace) -> MyNamespace:
    keys_to_process = [key for key in vars(parsed_args) if '.' in key]
    for key in keys_to_process:
        namespace, attr = key.split('.', 1)
        if hasattr(parsed_args, namespace):
            setattr(getattr(parsed_args, namespace), attr, getattr(parsed_args, key))
            delattr(parsed_args, key)
    return parsed_args


def parsers_parser(args: list=None) -> MyNamespace:
    parser = argparse.ArgumentParser()
    parser = add_lightning_trainer_argument(parser)
    parser = add_optimize_argument(parser)
    parser = add_model_config_argument(parser)
    # parser = add_model_init_argument(parser)
    parser = add_debug_argument(parser)
    parser = add_valid_argument(parser)
    parser = add_dataset_argument(parser)
    # parser.add_argument('--mode', type=str, default="train", choices=['train', 'eval', 'calculate'])

    parser_args: argparse.Namespace = parser.parse_args(args=args, namespace=MyNamespace())
    # 파싱 후 처리
    parser_args: MyNamespace = post_process_args(parsed_args=parser_args)
    
    # args post process

    NUM_THREADS = 32
    # parser_args.dataset.valid_task = parser_args.dataset.valid_task if parser_args.dataset.valid_task is not None else parser_args.dataset.train_task
    parser_args.dataset.train_num_workers = min(parser_args.trainer.train_batch_size, NUM_THREADS // parser_args.trainer.n_gpu)
    parser_args.dataset.valid_num_workers = min(parser_args.trainer.eval_batch_size, NUM_THREADS // parser_args.trainer.n_gpu)

    if parser_args.debug.test1000:  # debug option
        parser_args.debug.n_valid = 1024
        parser_args.debug.n_train = 8192
        parser_args.debug.n_test = 1024

    return parser_args


# if __name__ == '__main__':
#     args=[
#         '--trainer_seed', '42',
#         '--trainer_accelerator', 'gpu',
#         '--trainer_strategy', 'ddp',
#         '--trainer_precision', 'bf16-mixed',
#         '--trainer_n_gpu', '4',
#         '--trainer_train_batch_size', '128',
#         '--trainer_eval_batch_size', '1',
#         '--trainer_val_check_interval', '100',
#         '--trainer_log_root', 'logs',
#         '--trainer_output_dir', None,
#         '--trainer_init_ckpt', '',
#         '--optimize_weight_decay', '1e-4',
#         '--optimize_adam_epsilon', '1e-8',
#         '--optimize_max_grad_norm', '1.0',
#         '--optimize_learning_rate', '2e-4',
#         '--optimize_warmup_steps', '0',
#         '--optimize_num_train_epochs', '500',
#         '--optimize_gradient_accumulation_steps', '1',
#         '--model_config_plm_name_compressor', 'meta-llama/Llama-2-7b-hf',
#         '--model_config_plm_name_reader', 'meta-llama/Llama-2-7b-hf',
#         '--model_config_lora_r', '32',
#         '--model_config_lora_a', '32',
#         '--model_config_lora_target_modules', 'q_proj|v_proj|o_proj|k_proj|gate_proj|up_proj|down_proj',
#         '--model_config_num_layers', '32',
#         '--model_config_dynamic_mem_length', '0',
#         '--model_config_mem_max_len', '128',
#         '--debug_test1000', '0',
#         '--debug_n_train', '-1',
#         '--debug_n_valid', '-1',
#         '--debug_n_test', '-1',
#         '--valid_num_shot', '0',
#         '--dataset_train_task', 'instructfinetuning',
#         # '--dataset_valid_task', 'autoencoding',
#     ]
#     parser_args: MyNamespace = parsers_parser(args=args)
#     print(parser_args)
#     # print(parser_args.__dict__)
#     print(parser_args.trainer.__dict__)
#     print(parser_args.optimize.__dict__)
#     # print(parser_args.model_config.__dict__)
#     # print(parser_args.debug.__dict__)
#     # print(parser_args.valid.__dict__)
#     print(parser_args.dataset.__dict__)
