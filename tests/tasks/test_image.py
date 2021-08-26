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

import pytest
import torch
from flash.image import ImageClassificationData, ImageClassifier

from gradsflow.autotasks import AutoImageClassifier

datamodule = ImageClassificationData.from_folders(
    train_folder="data/hymenoptera_data/train/",
    val_folder="data/hymenoptera_data/val/",
)


def test_forward():
    model = AutoImageClassifier(datamodule)

    with pytest.raises(UserWarning):
        model.forward(torch.rand(1, 3, 8, 8))


def test_build_model():
    model = AutoImageClassifier(
        datamodule,
        max_epochs=1,
        timeout=5,
        suggested_backbones="ssl_resnet18",
        n_trials=1,
    )
    kwargs = {"backbone": "ssl_resnet18", "optimizer": "adam", "lr": 1e-1}
    model.model = model.build_model(**kwargs)
    assert isinstance(model.model, ImageClassifier)


def test_model():
    model = AutoImageClassifier(
        datamodule,
        max_epochs=1,
        timeout=5,
        suggested_backbones="ssl_resnet18",
        n_trials=1,
    )
    model.hp_tune()
