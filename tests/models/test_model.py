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
from pathlib import Path

import timm
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData

from gradsflow import AutoDataset, Model
from gradsflow.core.data import AutoDataset
from gradsflow.data.image import get_augmentations, image_dataset_from_directory
from gradsflow.models.model import Model
from gradsflow.models.tracker import Tracker

image_size = (96, 96)
transform = get_augmentations(image_size)
train_ds = FakeData(size=100, image_size=[3, *image_size], transform=transform)
val_ds = FakeData(size=100, image_size=[3, *image_size], transform=transform)
train_dl = DataLoader(train_ds)
val_dl = DataLoader(val_ds)

num_classes = train_ds.num_classes
autodataset = AutoDataset(train_dl, val_dl, num_classes=num_classes)

cnn = timm.create_model("ssl_resnet18", pretrained=False, num_classes=num_classes).eval()
model = Model(cnn)
model.compile("crossentropyloss", "adam")
model.TEST = True


def test_predict():
    x = torch.randn(1, 3, 64, 64)
    r1 = model.forward(x)
    r2 = model(x)
    r3 = model.predict(x)
    assert torch.all(torch.isclose(r1, r2))
    assert torch.all(torch.isclose(r2, r3))
    assert isinstance(model.predict(torch.randn(1, 3, 64, 64)), torch.Tensor)


def test_fit():
    tracker = model.fit(autodataset, epochs=10, steps_per_epoch=1)
    assert isinstance(tracker, Tracker)
