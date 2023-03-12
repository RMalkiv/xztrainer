from typing import Tuple, Dict, List

import torch
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics import Metric, Accuracy
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchvision.transforms import ToTensor

from xztrainer import XZTrainer, XZTrainerConfig, SchedulerType, XZTrainable, BaseContext, DataType, \
    ModelOutputType, ClassifierType, TrainContext
from xztrainer.logger.tensorboard import TensorboardLoggingEngineConfig
from xztrainer.setup_helper import set_seeds, enable_tf32

if __name__ == '__main__':
    set_seeds(0xCAFEBABE)
    enable_tf32()

    dataset_train = CIFAR10(root='./cifar10', download=True, train=True, transform=ToTensor())
    dataset_test = CIFAR10(root='./cifar10', download=True, train=False, transform=ToTensor())


    class SimpleTrainable(XZTrainable):
        def __init__(self):
            self.loss = CrossEntropyLoss()

        def step(self, context: BaseContext, data: DataType) -> Tuple[Tensor, Dict[str, ModelOutputType]]:
            img, label = data
            logits = context.model(img)
            preds = torch.argmax(logits, dim=1)
            loss = self.loss(logits, label)

            return loss, {'predictions': preds, 'targets': label}

        def on_load(self, context: TrainContext, step: int):
            print(f'Next step will be: {step}')

        def create_metrics(self) -> Dict[str, Metric]:
            return {
                'accuracy': Accuracy('multiclass', num_classes=10)
            }

        def update_metrics(self, model_outputs: Dict[str, List], metrics: Dict[str, Metric]):
            metrics['accuracy'].update(model_outputs['predictions'], model_outputs['targets'])

        def calculate_composition_metrics(self, metric_values: Dict[str, float]) -> Dict[str, float]:
            return {
                'accuracy_x2': metric_values['accuracy'] * 2
            }


    trainer = XZTrainer(
        config=XZTrainerConfig(
            batch_size=512,
            batch_size_eval=256,
            epochs=10,
            optimizer=lambda module: AdamW(module.parameters(), lr=1e-3, weight_decay=1e-4),
            amp_dtype=torch.float16,
            experiment_name='cifar10',
            gradient_clipping=1.0,
            scheduler=lambda optimizer, total_steps: OneCycleLR(optimizer, 1e-3, total_steps),
            scheduler_type=SchedulerType.STEP,
            save_steps=10,
            save_keep_n=3,
            dataloader_num_workers=8,
            accumulation_batches=4,
            print_steps=10,
            eval_steps=50,
            logger=TensorboardLoggingEngineConfig()
            # logger=ComposeLoggingEngineConfig(TensorboardLoggingEngineConfig(), StreamLoggingEngineConfig())
        ),
        model=resnet18(pretrained=False, num_classes=10),
        trainable=SimpleTrainable()
    )
    trainer.train(dataset_train, dataset_test)
