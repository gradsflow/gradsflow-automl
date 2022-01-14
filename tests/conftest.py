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
# Arrange
from pathlib import Path

import pytest
import timm
from torch import nn

from gradsflow import AutoDataset, Model
from gradsflow.data import image_dataset_from_directory
from gradsflow.models.tracker import Tracker

data_dir = Path.cwd()
folder = f"{data_dir}/data/test-data-cat-dog-v0/cat-dog/"
data = image_dataset_from_directory(folder, transform=True, ray_data=False)


@pytest.fixture
def auto_dataset():
    return AutoDataset(train_dataloader=data.dataloader, val_dataloader=data.dataloader)


@pytest.fixture
def resnet18():
    cnn = timm.create_model("ssl_resnet18", pretrained=False, num_classes=10).eval()

    return cnn


@pytest.fixture
def cnn_model(resnet18):
    model = Model(resnet18)
    model.TEST = True

    return model


@pytest.fixture
def tracker():
    return Tracker()


@pytest.fixture
def dummy_model():
    """A dummy torch.nn model that adds 1 to the forward input value."""

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x + 1

    return DummyModel()
