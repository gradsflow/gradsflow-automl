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

import pytest
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset

from gradsflow.data import AutoDataset

dataset = TensorDataset(torch.randn(8, 1, 32, 32))
dataloader = DataLoader(dataset)

from gradsflow.data.image import image_dataset_from_directory

data_dir = Path.cwd()
folder = f"{data_dir}/data/test-data-cat-dog-v0/cat-dog/"
data = image_dataset_from_directory(folder, transform=True, ray_data=False)


def test_auto_dataset():
    with pytest.raises(UserWarning):
        AutoDataset()


def test_sent_to_device():
    accelerate = Accelerator()
    autodata = AutoDataset(dataloader)
    assert autodata.device_setup_status is None
    autodata.prepare_data(accelerate)
    assert autodata.device_setup_status


def test_dataset():
    accelerate = Accelerator()
    autodata = AutoDataset(train_dataset=data.dataset, val_dataset=data.dataset)
    autodata.prepare_data(accelerate)
    assert isinstance(autodata.train_dataloader, DataLoader)
