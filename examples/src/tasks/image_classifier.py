#  Copyright (c) 2021 GradsFlow. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as T

from gradsflow import AutoImageClassifier
from gradsflow.data.common import random_split_dataset

# Replace dataloaders with your custom dataset and you are all set to train your model
image_size = (64, 64)
batch_size = 4

to_rgb = lambda x: x.convert("RGB")

# TODO: Add argument parser
if __name__ == "__main__":
    augs = T.Compose([to_rgb, T.AutoAugment(), T.Resize(image_size), T.ToTensor()])
    data = torchvision.datasets.CIFAR10("~/data", download=True, transform=augs)
    train_data, val_data = random_split_dataset(data, 0.01)
    train_dl = DataLoader(train_data, batch_size=batch_size)
    val_dl = DataLoader(val_data, batch_size=batch_size)

    num_classes = len(data.classes)

    model = AutoImageClassifier(
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        num_classes=num_classes,
        max_epochs=5,
        optimization_metric="train_loss",
        max_steps=1,
        n_trials=1,
    )
    print("AutoImageClassifier initialised!")

    model.hp_tune()
