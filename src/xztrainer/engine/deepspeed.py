# import json
# import logging
# import tempfile
# from argparse import Namespace
# from dataclasses import dataclass
#
# import deepspeed
#
# from .. import SchedulerType
#
#
# @dataclass
# class DeepSpeedConfig(XZTrainerEngineConfig):
#     fp16: bool = True
#     zero: bool = True
#
#     def create_engine(self, trainer) -> XZTrainerEngine:
#         return DeepSpeedEngine(trainer, self)
#
#
# class DeepSpeedEngine(XZTrainerEngine):
#     def __init__(self, trainer, config: DeepSpeedConfig):
#         super().__init__(trainer)
#         self.config = config
#
#     def _write_deepspeed_cfg(self) -> str:
#         cfg = self.trainer.config
#         cfgds = self.config
#         deepspeed_cfg = {  # todo play with this stuff a bit
#             'zero_allow_untested_optimizer': True,
#             'train_batch_size': cfg.batch_size * cfg.accumulation_steps,
#             'train_micro_batch_size_per_gpu': cfg.batch_size,
#             "fp16": {
#                 "enabled": cfgds.fp16,
#                 "loss_scale": 0,
#                 "initial_scale_power": 32,
#                 "loss_scale_window": 1000,
#                 "hysteresis": 2,
#                 "min_loss_scale": 1
#             },
#             'gradient_clipping': cfg.gradient_clipping,
#             "zero_optimization": {
#                 "stage": 2 if cfgds.zero else 0,
#             }
#
#         }
#         with tempfile.NamedTemporaryFile('w', delete=False) as f:
#             logging.info(f'Writing DeepSpeed configuration into {f.name}')
#             json.dump(deepspeed_cfg, f)
#             return f.name
#
#     def wrap_model(self, model, optim, scheduler, scheduler_type):
#         deepspeed_cfg = self._write_deepspeed_cfg()
#         model, optim, scheduler_, _ = deepspeed.initialize(
#             args=Namespace(**{'local_rank': 0, 'deepspeed_config': deepspeed_cfg}),
#             model=model,
#             optimizer=optim,
#             model_parameters=model.parameters(),
#             training_data=None,
#             lr_scheduler=scheduler if scheduler_type == SchedulerType.STEP else None
#         )
#         if scheduler_ is not None:
#             scheduler = scheduler_
#         return model, optim, scheduler
#
#     def backward_pass(self, do_train, model, optimizer, scheduler, i, loss):
#         model.backward(loss)
#         if do_train:
#             model.step()
