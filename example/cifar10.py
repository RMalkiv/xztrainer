from typing import Tuple, Dict, List

import torch
from accelerate import Accelerator
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset
from torchmetrics import Metric, Accuracy
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchvision.transforms import ToTensor

from xztrainer import XZTrainer, XZTrainerConfig, XZTrainable, BaseContext, DataType, \
    ModelOutputType, TrainContext, ContextType, TrackerConfigType
from xztrainer.setup_helper import set_seeds, enable_tf32


class SimpleTrainable(XZTrainable):
    def __init__(self):
        self.loss = CrossEntropyLoss()

    def step(self, context: BaseContext, data: DataType) -> Tuple[Tensor, Dict[str, ModelOutputType]]:
        logits = context.model(data['image'])
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, data['label'])

        return loss, {'predictions': preds, 'targets': data['label']}

    def on_load(self, context: TrainContext, step: int):
        print(f'Next step will be: {step}')

    def create_metrics(self, context_type: ContextType) -> Dict[str, Metric]:
        return {
            'accuracy': Accuracy('multiclass', num_classes=10)
        }

    def update_metrics(self, context_type: ContextType, model_outputs: Dict[str, List], metrics: Dict[str, Metric]):
        metrics['accuracy'].update(model_outputs['predictions'], model_outputs['targets'])

    def calculate_composition_metrics(self, context_type: ContextType, metric_values: Dict[str, float]) -> Dict[
        str, float]:
        return {
            'accuracy_x2': metric_values['accuracy'] * 2
        }

    def tracker_config(self, context: TrainContext) -> TrackerConfigType:
        return {
            'comment': 'my first training'
        }


class CifarDictDataset(Dataset):
    def __init__(self, train: bool):
        self.base_data = CIFAR10(root='./cifar10', download=True, train=train, transform=ToTensor())

    def __getitem__(self, item):
        image, label = self.base_data[item]
        return {
            'image': image,
            'label': torch.scalar_tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.base_data)


class CifarCollator:
    def __call__(self, batch: list):
        return {
            'image': torch.stack([x['image'] for x in batch], dim=0),
            'label': torch.stack([x['label'] for x in batch], dim=0)
        }


if __name__ == '__main__':
    set_seeds(0xCAFEBABE)
    enable_tf32()

    dataset_train = CifarDictDataset(train=True)
    dataset_test = CifarDictDataset(train=False)

    accelerator = Accelerator(
        gradient_accumulation_steps=32,
        log_with='tensorboard',
        project_dir='.'
    )

    config = XZTrainerConfig(
        experiment_name='exp-2',
        minibatch_size=32,
        minibatch_size_eval=256,
        epochs=10,
        optimizer=lambda module: AdamW(module.parameters(), lr=1e-3, weight_decay=1e-4),
        gradient_clipping=1.0,
        scheduler=lambda optimizer, total_steps: OneCycleLR(optimizer, 1e-3, total_steps),
        save_steps=100,
        dataloader_persistent_workers=True,
        dataloader_num_workers=8,
        log_steps=10,
        eval_steps=500,
        tracker_config={'model_revision': 'resnet18'},
        collate_fn=CifarCollator()
    )

    model = resnet18(weights=None, num_classes=10)

    trainer = XZTrainer(
        config=config,
        model=model,
        trainable=SimpleTrainable(),
        accelerator=accelerator
    )
    trainer.train(dataset_train, dataset_test)
