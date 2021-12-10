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
import os
from pathlib import Path

import pytest

from gradsflow import AutoDataset
from gradsflow.callbacks import EmissionTrackerCallback
from gradsflow.callbacks.logger import CometCallback, CSVLogger
from gradsflow.data.image import image_dataset_from_directory
from gradsflow.utility.imports import is_installed
from tests.dummies import DummyModel

data_dir = Path.cwd()
folder = f"{data_dir}/data/test-data-cat-dog-v0/cat-dog/"
data = image_dataset_from_directory(folder, transform=True, ray_data=False)


@pytest.fixture
def dummy_model():
    return DummyModel()


@pytest.fixture
def auto_dataset():
    return AutoDataset(train_dataloader=data.dataloader, val_dataloader=data.dataloader)


def test_csv_logger(dummy_model, auto_dataset):
    csv_logger = CSVLogger(filename="test_csv_logger.csv")
    dummy_model.compile()
    dummy_model.fit(auto_dataset, callbacks=csv_logger)
    assert os.path.isfile("test_csv_logger.csv")


@pytest.mark.skipif(is_installed("comet"), reason="requires `comet_ml` installed")
def test_comet(dummy_model, auto_dataset):
    with pytest.raises(ValueError):
        CometCallback()

    comet = CometCallback(offline=True)
    dummy_model.compile()
    dummy_model.fit(auto_dataset, callbacks=[comet])


@pytest.mark.skipif(is_installed("codecarbon"), reason="requires `codecarbon` installed")
def test_emission_tracker(dummy_model, auto_dataset):
    emission_tracker = EmissionTrackerCallback()
    dummy_model.compile()
    dummy_model.fit(auto_dataset, callbacks=[emission_tracker])
