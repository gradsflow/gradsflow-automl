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

import argparse

import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms as T

from gradsflow import AutoImageClassifier
from gradsflow.data.common import random_split_dataset


def main(image_size=(64, 64), batch_size=4, max_epochs=5, optimization_metric="train_loss", max_steps=1, n_trials=1):
    # Replace dataloaders with your custom dataset and you are all set to train your model

    to_rgb = lambda x: x.convert("RGB")
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
        max_epochs=max_epochs,
        optimization_metric=optimization_metric,
        max_steps=max_steps,
        n_trials=n_trials,
    )
    print("AutoImageClassifier initialised!")

    model.hp_tune()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoImageClassifier")
    parser.add_argument("--image_size", type=tuple, default=(64, 64), help="image size")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--max_epochs", type=int, default=5, help="max epochs")
    parser.add_argument("--optimization_metric", type=str, default="train_loss", help="optimization metric")
    parser.add_argument("--max_steps", type=int, default=1, help="max steps")
    parser.add_argument("--n_trials", type=int, default=1, help="number of trials")
    args = parser.parse_args()
    main(**vars(args))
