# Quick Start

## What is xztrainer

It is just a minimalistic PyTorch [training loop implementation](https://xkcd.com/927/) using open-source packages such as `torchmetrics` and `accelerate` to tackle with all the heavy lifting. 
No complex ecosystem, no bloatware, no "what the hell does that 5k lines of code function do?". Read the docs (and optionally code), implement what you need and start making fun training your models.

## How-To

### 1. Install xztrainer

Run `pip install xztrainer torch torchmetrics accelerate` or `poetry add xztrainer torch torchmetrics accelerate` depending on package manager you use.

### 2. Set up Accelerate

Since xztrainer uses Accelerate, you need to [configure the training setup](https://huggingface.co/docs/accelerate/en/quicktour#unified-launch-interface).

* To save configuration globally - just launch `accelerate config` and specify parameters for your training setup.
* To save configuration locally - same as above, but also specify `--config-file` argument: `accelerate config --config-file .accelerate`

### 3. Write your training script

See an example [here](https://github.com/mrapplexz/xztrainer/blob/develop/example/cifar10.py).

#### 3.1. Prepare

Specify some preparation steps in your code, such as locking random seeds and enabling TF32 computations. See [xztrainer utilities docs](utilities.md).

```python
from xztrainer.setup_helper import set_seeds, enable_tf32

set_seeds(0xCAFEBABE)
enable_tf32()
```

#### 3.2. Implement your Trainable

You need to implement your custom training logic by subclassing a `XZTrainable` class.

Trainable is used for:
* forward pass code, including loss computation;
* specifying what metrics you want to calculate while training (xztrainer uses torchmetrics for updating and calculating the metrics since it supports distributed metric computation out of the box, see [torchmetrics](https://lightning.ai/docs/torchmetrics/stable/) docs);
* specifying some other callbacks, such as model loading callback or logging callback

Use [xztrainer trainable docs](trainable.md) to see full list of functions you can implement in your Trainable.

An example that uses cross-entropy loss for an image classification model, calculating accuracy as a metric:

```python
from xztrainer import XZTrainable, BaseContext, DataType, ContextType, ModelOutputType
import torch
from torch import nn
import torchmetrics

class SimpleTrainable(XZTrainable):
    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    def step(self, context: BaseContext, data: DataType) -> tuple[torch.Tensor, dict[str, ModelOutputType]]:
        logits = context.model(data['image'])
        preds = torch.argmax(logits, dim=1)
        loss = self.loss(logits, data['label'])

        return loss, {'predictions': preds, 'targets': data['label']}

    def create_metrics(self, context_type: ContextType) -> dict[str, torchmetrics.Metric]:
        return {
            'accuracy': torchmetrics.Accuracy('multiclass', num_classes=10)
        }

    def update_metrics(
            self, 
            context_type: ContextType, 
            model_outputs: dict[str, ModelOutputType], 
            metrics: dict[str, torchmetrics.Metric]
    ):
        metrics['accuracy'].update(model_outputs['predictions'], model_outputs['targets'])
```


#### 3.3. Create standard PyTorch objects

You need to implement your standard PyTorch objects related to working with data. And, for sure, your model object.

##### Dataset

An example [dataset](https://pytorch.org/docs/stable/data.html#dataset-types) that remaps standard torchvision CIFAR10 dataset from tuple-yielding to dictionary-yielding just for convenience.

```python
import torch
from torch.utils.data import Dataset
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import ToTensor

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
```

##### Collator

An example [collate function](https://pytorch.org/data/main/generated/torchdata.datapipes.iter.Collator.html) that stacks images and labels

```python
import torch

class CifarCollator:
    def __call__(self, batch: list):
        return {
            'image': torch.stack([x['image'] for x in batch], dim=0),
            'label': torch.stack([x['label'] for x in batch], dim=0)
        }
```

##### Model

We will just use resnet18 model from torchvision.

```python
from torchvision.models.resnet import resnet18

model = resnet18(weights=None, num_classes=10)
```


#### 3.4. Create training configuration

You need to create a configuration for the trainer.

Use [xztrainer trainer documentation](trainer.md) to see what parameters you can configure.

Example:

```python
from xztrainer import XZTrainerConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

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
    collate_fn=CifarCollator()
)
```


#### 3.5. Initialize Accelerator object

Since xztrainer uses Accelerate internally, you need to create an Accelerator object.

Use [Accelerator documentation](https://huggingface.co/docs/accelerate/en/package_reference/accelerator) to see what parameters you can configure here.

Example with gradient accumulation and TensorBoard logging:

```python
from accelerate import Accelerator

accelerator = Accelerator(
    gradient_accumulation_steps=32,
    log_with='tensorboard',
    project_dir='./training-logs/'
)
```

#### 3.6. Start training

Instantiate `XZTrainer` instance with all the objects you created before and call `train(...)` function.

Use [xztrainer trainer documentation](trainer.md) to see what happens inside `train(...)` function.


Example:

```python
from xztrainer import XZTrainer

trainer = XZTrainer(
    config=config,
    model=model,
    trainable=SimpleTrainable(),
    accelerator=accelerator
)
trainer.train(train_data=CifarDictDataset(train=True), eval_data=CifarDictDataset(train=False))
```



### 4. Run your training script using Accelerate

* If running a Python module - `accelerate launch -m mymodule.train` or `accelerate launch -m mymodule.train --config-file .accelerate` if you have a local configuration file
* If running a Python script - `accelerate launch mymodule/train.py` or `accelerate launch mymodule/train.py --config-file .accelerate` if you have a local configuration file

### 5. Explore saved artifacts

Inside a `project_dir` you specified in a `Accelerator` configuration, you will see:

* Saved checkpoints inside `checkpoint` directory
* In case of logging enabled - logging artifacts in `runs` directory