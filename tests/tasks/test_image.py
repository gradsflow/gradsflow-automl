# Copyright (c) 2021 GradsFlow. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

os.environ["GF_CI"] = "true"

import warnings
from pathlib import Path

import pytest
import torch

from gradsflow.data.image import image_dataset_from_directory
from gradsflow.models.model import Model
from gradsflow.tasks import AutoImageClassifier

warnings.filterwarnings("ignore")

data_dir = Path.cwd() / "data"

train_data = image_dataset_from_directory(f"{data_dir}/hymenoptera_data/train/", transform=True)
train_dl = train_data["dl"]

val_data = image_dataset_from_directory(f"{data_dir}/hymenoptera_data/val/", transform=True)
val_dl = val_data["dl"]


def test_forward():
    model = AutoImageClassifier(
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        num_classes=2,
    )

    with pytest.raises(UserWarning):
        model.forward(torch.rand(1, 3, 8, 8))


def test_build_model():
    automodel = AutoImageClassifier(
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        num_classes=2,
        max_epochs=1,
        timeout=5,
        suggested_backbones="ssl_resnet18",
        n_trials=1,
    )
    kwargs = {"backbone": "ssl_resnet18", "optimizer": "adam", "lr": 1e-1}
    automodel.model = automodel.build_model(kwargs)
    assert isinstance(automodel.model, Model)


def test_hp_tune():
    model = AutoImageClassifier(
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        num_classes=2,
        max_epochs=1,
        max_steps=2,
        timeout=30,
        suggested_backbones="ssl_resnet18",
        optimization_metric="val_accuracy",
        n_trials=1,
    )
    model.hp_tune(name="pytest-experiment", mode="max", gpu=0)
