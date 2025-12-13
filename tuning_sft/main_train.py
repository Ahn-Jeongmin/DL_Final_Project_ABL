import os
import time
import torch
import lightning.pytorch as L
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.plugins.precision import DeepSpeedPrecision
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pl_module import MyLightningModule
from arg_parser import MyNamespace, parsers_parser
from datamodules import VQADataModule

print(torch.__version__)
print(L.__version__)

logger = None

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))


def train(args: MyNamespace):
    module = MyLightningModule(args=args)
    datamodule = VQADataModule(args=args)

    lr_monitor = LearningRateMonitor()
    checkpoint_callbacks = []
    checkpoint_callbacks.append(ModelCheckpoint(
        dirpath=args.trainer.output_dir,
        # filename=args.tag_info + '_epoch={epoch}-avg_train_loss={avg_train_loss:.6f}',
        filename='_epoch={epoch}-step={step}-avg_train_loss={avg_train_loss:.6f}',
        monitor="avg_train_loss",
        # verbose: bool = False,
        # save_last: bool | None = None,
        save_top_k=3,
        # save_weights_only: bool = False,
        mode="min",
        auto_insert_metric_name=False,
        # every_n_train_steps: int | None = None,
        # train_time_interval: timedelta | None = None,
        # every_n_epochs=args.check_val_every_n_epoch,
        save_on_train_epoch_end=True,
    ))
    metric_names = []
    # metric_names.append('EM')
    # metric_names.append('F1')
    # metric_names.append('rouge1')
    # metric_names.append('rougeL')
    metric_names.append('rougeL_all')
    metric_names.append('rougeL_ambig')
    metric_names.append('rougeL_unambig')
    metric_names.append('rta_all')
    metric_names.append('rta_ambig')
    metric_names.append('rta_unambig')

    for mn in metric_names:
        checkpoint_callbacks.append(ModelCheckpoint(
            dirpath=args.trainer.output_dir,
            # filename=args.tag_info + '_epoch={epoch}-recall@1={metric/ppl:.6f}',
            filename='_epoch={epoch}-step={step}-' + f"{mn}" + '={metric/' + f"{mn}" + ':.6f}',
            monitor=f"metric/{mn}",
            # verbose: bool = False,
            # save_last: bool | None = None,
            save_top_k=2,
            save_weights_only = True,
            mode="max",
            auto_insert_metric_name=False,
            # every_n_train_steps: int | None = None,
            # train_time_interval: timedelta | None = None,
            # every_n_epochs=args.check_val_every_n_epoch,
            save_on_train_epoch_end=False,
        ))

    if 'zero' in args.trainer.strategy:
        #region deepspeed options
        ### deepspeed options
        # config: _PATH | Dict[str, Any] | None = None,

        ## env options
        # accelerator: Any | None = None,
        # precision_plugin: Precision | None = None,

        ## zero options
        # zero_optimization: bool = True,
        # stage: int = 2,
        # remote_device: str | None = None,
        # offload_optimizer: bool = False,
        # offload_optimizer_device: str = "cpu",
        # contiguous_gradients: bool = True,
        # overlap_comm: bool = True,
        # allgather_partitions: bool = True,
        # reduce_scatter: bool = True,
        # allgather_bucket_size: int = 200000000,
        # reduce_bucket_size: int = 200000000,
        # zero_allow_untested_optimizer: bool = True,

        ## ZeRO stage 3 options
        # offload_parameters: bool = False,
        # offload_params_device: str = "cpu",
        # pin_memory: bool = False,
        # sub_group_size: int = 1000000000000,
        # partition_activations: bool = False,
        # cpu_checkpointing: bool = False,
        # load_full_weights: bool = False,

        ## nvme options
        # nvme_path: str = "/local_nvme",
        # params_buffer_count: int = 5,
        # params_buffer_size: int = 100000000,
        # max_in_cpu: int = 1000000000,
        # optimizer_buffer_count: int = 4,
        # block_size: int = 1048576,
        # queue_depth: int = 8,
        # single_submit: bool = False,
        # overlap_events: bool = True,
        # thread_count: int = 1,

        ## FP16 training options
        # loss_scale: float = 0,
        # loss_scale_window: int = 1000,
        # initial_scale_power: int = 16,
        # hysteresis: int = 2,
        # min_loss_scale: int = 1,\

        ## other options
        # logging_batch_size_per_gpu: str | int = "auto",
        # logging_level: int = logging.WARN,
        # contiguous_memory_optimization: bool = False,
        # synchronize_checkpoint_boundary: bool = False,
        # parallel_devices: List[device] | None = None,
        # cluster_environment: ClusterEnvironment | None = None,
        # process_group_backend: str | None = None
        
        #endregion

        accelerator = 'auto'
        _accelerator = args.trainer.accelerator
        _precision_plugin = DeepSpeedPrecision(precision=args.trainer.precision)
        if 'zero_1' in args.trainer.strategy:
            # shard optimizer_state
            _stage = 1
        elif 'zero_2' in args.trainer.strategy:
            # shard optimizer_state & gradients
            _stage = 2
        elif 'zero_3' in args.trainer.strategy:
            # shard optimizer_state & gradients & weights
            _stage = 3
            raise NotImplementedError("stage 3 not supported")
        else:
            raise ValueError(f"strategy '{args.trainer.strategy}' not supported")
        _offload_optimizer = '_offload_opt' in args.trainer.strategy
        # _offload_parameters = '_offload_param' in args.trainer.strategy

        strategy = DeepSpeedStrategy(
            ## env options
            accelerator=_accelerator,
            precision_plugin=_precision_plugin,

            ## ZeRO stage 1, 2 options
            zero_optimization=True,
            stage=_stage,
            offload_optimizer=_offload_optimizer,
            # allgather_bucket_size: int = 200000000,
            # reduce_bucket_size: int = 200000000,
            # zero_allow_untested_optimizer: bool = True,
        
            ## ZeRO stage 3 options
            # offload_parameters=_offload_parameters,
        )
    else:
        # strategy = DDPStrategy(
        #     accelerator=args.accelerator,
        #     # parallel_devices: List[device] | None = None,
        #     # cluster_environment: ClusterEnvironment | None = None,
        #     # checkpoint_io: CheckpointIO | None = None,
        #     # precision_plugin: PrecisionPlugin | None = None,
        #     # ddp_comm_state: object | None = None,
        #     # ddp_comm_hook: ((...) -> Any) | None = None,
        #     # ddp_comm_wrapper: ((...) -> Any) | None = None,
        #     # model_averaging_period: int | None = None,
        #     # process_group_backend: str | None = None,
        #     # timeout: timedelta | None = default_pg_timeout,
        #     # start_method: Literal['popen', 'spawn', 'fork', 'forkserver'] = "popen",
        #     # **kwargs: Any
        # )
        accelerator = args.trainer.accelerator
        strategy = args.trainer.strategy

    # trainer_params = dict(
    #     accelerator=args.accelerator,
    #     strategy=strategy,
    #     devices=args.n_gpu,
    #     num_nodes=1,
    #     precision=args.precision,
    #     logger=logger,
    #     callbacks=[
    #         lr_monitor, checkpoint_callback_1,
    #         # checkpoint_callback_2, checkpoint_callback_3
    #     ],
    #     # fast_dev_run: int | bool = False,
    #     max_epochs=args.num_train_epochs,
    #     # min_epochs: int | None = None,
    #     # max_steps: int = -1,
    #     # min_steps: int | None = None,
    #     # max_time: str | timedelta | Dict[str, int] | None = None,
    #     # limit_train_batches: int | float | None = None,
    #     # limit_val_batches: int | float | None = None,
    #     # limit_test_batches: int | float | None = None,
    #     # limit_predict_batches: int | float | None = None,
    #     # overfit_batches: int | float = 0,
    #     # val_check_interval=args.val_check_interval,
    #     # check_val_every_n_epoch=args.check_val_every_n_epoch,
    #     # num_sanity_val_steps=2,  # 2
    #     # log_every_n_steps=50,
    #     # enable_checkpointing: bool | None = None,
    #     # enable_progress_bar: bool | None = None,
    #     # enable_model_summary: bool | None = None,
    #     accumulate_grad_batches=args.gradient_accumulation_steps,
    #     gradient_clip_val=args.max_grad_norm,
    #     # gradient_clip_algorithm: str | None = None,
    #     # deterministic: bool | Literal['warn'] | None = None,
    #     # benchmark: bool | None = None,
    #     # inference_mode: bool = True,
    #     # use_distributed_sampler: bool = True,
    #     # profiler: Profiler | str | None = None,
    #     detect_anomaly=False,
    #     # detect_anomaly=True,
    #     barebones=False,
    #     # plugins: PLUGIN_INPUT | List[PLUGIN_INPUT] | None = None,
    #     # sync_batchnorm: bool = False,
    #     # reload_dataloaders_every_n_epochs: int = 0,
    #     # default_root_dir: _PATH | None = None
    # )

    trainer = L.Trainer(
        accelerator=accelerator,
        strategy=strategy,
        devices=args.trainer.n_gpu,
        num_nodes=1,
        precision=args.trainer.precision,
        logger=logger,
        callbacks=[lr_monitor,] + checkpoint_callbacks,
        # fast_dev_run: int | bool = False,
        max_epochs=args.optimize.num_train_epochs,
        # min_epochs: int | None = None,
        # max_steps: int = -1,
        # min_steps: int | None = None,
        # max_time: str | timedelta | Dict[str, int] | None = None,
        # limit_train_batches: int | float | None = None,
        # limit_val_batches: int | float | None = None,
        # limit_test_batches: int | float | None = None,
        # limit_predict_batches: int | float | None = None,
        # overfit_batches: int | float = 0,
        val_check_interval=args.trainer.val_check_interval,
        # check_val_every_n_epoch=args.check_val_every_n_epoch,
        num_sanity_val_steps=0,  # 2
        # num_sanity_val_steps=2,  # 2
        log_every_n_steps=1,
        enable_checkpointing=True,
        # enable_progress_bar: bool | None = None,
        # enable_model_summary: bool | None = None,
        accumulate_grad_batches=args.optimize.gradient_accumulation_steps,
        gradient_clip_val=args.optimize.max_grad_norm,
        # gradient_clip_algorithm: str | None = None,
        # deterministic: bool | Literal['warn'] | None = None,
        # benchmark: bool | None = None,
        # inference_mode: bool = True,
        # use_distributed_sampler: bool = True,
        # profiler: Profiler | str | None = None,
        detect_anomaly=False,
        # detect_anomaly=True,
        barebones=False,
        # plugins: PLUGIN_INPUT | List[PLUGIN_INPUT] | None = None,
        # sync_batchnorm: bool = False,
        # reload_dataloaders_every_n_epochs: int = 0,
        # default_root_dir: _PATH | None = None  # Default path for logs and weights when no logger/ckpt_callback passed.
    )
    
    trainer.test(
        model=module,
        datamodule=datamodule,
    )
    trainer.validate(
        model=module,
        datamodule=datamodule,
    )

    trainer.fit(
        model=module,
        datamodule=datamodule,
        # model: LightningModule,
        # train_dataloaders: TRAIN_DATALOADERS | LightningDataModule | None = None,
        # val_dataloaders: EVAL_DATALOADERS | None = None,
        # datamodule: LightningDataModule | None = None,
        # ckpt_path: str | None = Nonemodel
    )
    trainer.test(
        model=module,
        datamodule=datamodule,
    )
    # trainer.validate(module)
    # trainer.validate(
    #     model=model,
    #     # ckpt_path="logs_1/checkpoints/version_3/_epoch=3-step=7360-NQ_F1=57.745705.ckpt",
    #     ckpt_path="logs_1/checkpoints/version_7/_epoch=3-step=6900-NQ_F1=51.695885.ckpt",
    #     # verbose: bool = True,
    #     # datamodule: LightningDataModule | None = None
    # )
    # trainer.validate(model, ckpt_path='logs_4/checkpoints/version_6/nq:_1_trivia:_0_kilt:_0_gtq_doc_aug_qg_t5-base_dev_recall_adanum:_4_dropgtqdocindev:_0_lre0.4d2.9999999999999996_epoch=19-avg_train_loss=0.112345.ckpt')
    # trainer.validate(model, ckpt_path='logs_4/checkpoints/version_7/nq:_1_trivia:_0_kilt:_0_gtq_doc_aug_qg_t5-base_dev_recall_adanum:_4_dropgtqdocindev:_0_lre0.4d2.9999999999999996_epoch=19-avg_train_loss=0.099478.ckpt')


if __name__ == "__main__":
    # testargs = [
    #     '--seed', 42,
    #     '--strategy', 'ddp',
    #     '--n_gpu', 1,
    #     '--train_batch_size', 1,
    #     '--learning_rate', 2e-5,
    #     '--warmup_steps', 1000,
    #     '--num_train_epochs', 20,
    #     '--gradient_accumulation_steps', 1,
    #     '--test1000', 0,
    #     '--log_root', 'logs_test',
    # ]
    # testargs = [str(arg) for arg in testargs]
    testargs = None
    args = parsers_parser(args=testargs)

    L.seed_everything(seed=args.trainer.seed)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(dir_path)
    print(parent_path)
    args.trainer.logs_dir = os.path.join(dir_path, args.trainer.log_root)

    # this is model pkl save dir
    args.trainer.output_dir = os.path.join(dir_path, args.trainer.log_root)

    time_str = time.strftime("%Y%m%d-%H%M%S")

    # logger = TensorBoardLogger(f"{args.trainer.log_root}/", name='default', version=269)
    # logger = TensorBoardLogger(f"{args.trainer.log_root}/", name='default', version=None)
    logger = TensorBoardLogger(f"{args.trainer.log_root}/", name='default', version=args.debug.log_version)
    args.trainer.output_dir = os.path.join(args.trainer.output_dir, 'checkpoints', f'version_{logger.version}')
    ###########################

    train(args)
