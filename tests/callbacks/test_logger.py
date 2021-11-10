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
from gradsflow.callbacks.logger import CometCallback, CSVLogger
from gradsflow.data.image import image_dataset_from_directory
from tests.dummies import DummyModel

data_dir = Path.cwd()
folder = f"{data_dir}/data/test-data-cat-dog-v0/cat-dog/"
data = image_dataset_from_directory(folder, transform=True, ray_data=False)


def test_csv_logger():
    csv_logger = CSVLogger(filename="test_csv_logger.csv")
    autodata = AutoDataset(train_dataloader=data.dataloader, val_dataloader=data.dataloader)
    model = DummyModel()
    model.compile()
    model.fit(autodata, callbacks=csv_logger)
    assert os.path.isfile("test_csv_logger.csv")


def test_logger():
    with pytest.raises(ValueError):
        CometCallback()

    comet = CometCallback(offline=True)
    autodata = AutoDataset(train_dataloader=data.dataloader, val_dataloader=data.dataloader)
    model = DummyModel()
    model.compile()
    model.fit(autodata, callbacks=[comet])
