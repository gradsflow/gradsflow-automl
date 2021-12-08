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
import dataclasses
from typing import Union

from torch.utils.data import DataLoader, Dataset

from gradsflow.data.ray_dataset import RayDataset


@dataclasses.dataclass(init=False)
class Data:
    dataloader: DataLoader
    dataset: Union[RayDataset, Dataset]


class BaseAutoDataset:
    def __init__(self):
        self.meta = {}
        self.datamodule = None
        self._train_dataloader = None
        self._val_dataloader = None
        self.train_dataset = None
        self.val_dataset = None
        self.num_classes = None
