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
from unittest.mock import MagicMock, patch

import pytest
import torch
from flash.image import ImageClassificationData

from gradsflow.autotasks.engine.automodel import AutoModel
from gradsflow.autotasks.engine.backend import BackendType

cwd = Path.cwd()
datamodule = ImageClassificationData.from_folders(
    train_folder=f"{cwd}/data/hymenoptera_data/train/", val_folder=f"{cwd}/data/hymenoptera_data/val/", batch_size=1
)


@patch.multiple(AutoModel, __abstractmethods__=set())
def test_auto_model():
    assert AutoModel(datamodule)


@patch.multiple(AutoModel, __abstractmethods__=set())
def test_build_model():
    model = AutoModel(datamodule)
    with pytest.raises(NotImplementedError):
        model.build_model({"lr": 1})


@patch.multiple(AutoModel, __abstractmethods__=set())
def test_create_search_space():
    model = AutoModel(datamodule)
    with pytest.raises(NotImplementedError):
        model._create_search_space()


@patch.multiple(AutoModel, __abstractmethods__=set())
@patch("gradsflow.autotasks.engine.backend.FlashTrainer")
@patch("gradsflow.autotasks.engine.backend.PLTrainer")
def test_objective(mock_pl_trainer, mock_fl_trainer):
    optimization_metric = "val_accuracy"
    model = AutoModel(
        datamodule,
        optimization_metric=optimization_metric,
        backend=BackendType.pl.value,
    )

    model.backend.model_builder = MagicMock()

    mock_pl_trainer.callback_metrics = mock_fl_trainer.callback_metrics = {optimization_metric: torch.as_tensor([1])}

    model.backend.optimization_objective({}, {})


@patch.multiple(AutoModel, __abstractmethods__=set())
@patch("gradsflow.autotasks.engine.automodel.tune.run")
def test_hp_tune(
    mock_tune,
):
    automodel = AutoModel(datamodule)
    automodel._create_search_space = MagicMock()
    automodel.build_model = MagicMock()
    automodel.hp_tune(gpu=0, cpu=2)

    mock_tune.assert_called()
