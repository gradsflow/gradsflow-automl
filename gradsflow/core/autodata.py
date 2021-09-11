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
from typing import List, Optional

import pytorch_lightning as pl
import ray
from loguru import logger
from torch.utils.data import IterableDataset


class AutoDataset:
    def __init__(
        self,
        train_dataloader: Optional = None,
        val_dataloader: Optional = None,
        datamodule: Optional[pl.LightningDataModule] = None,
        num_classes: Optional[int] = None,
    ):

        self.datamodule = None
        self.num_classes = num_classes

        if not (datamodule or train_dataloader):
            raise UserWarning("Both datamodule and train_dataloader can't be None!")

        if all((datamodule, train_dataloader)):
            logger.warning(
                "Both datamodule and train_dataloader is set!"
                "Using datamodule over train_dataloader."
            )

        if not datamodule:
            datamodule = pl.LightningDataModule()
            datamodule.train_dataloader = train_dataloader
            datamodule.val_dataloader = val_dataloader
            datamodule.num_classes = num_classes

        if datamodule:
            self.datamodule = datamodule
            if hasattr(datamodule, "num_classes"):
                num_classes = datamodule.num_classes
            if num_classes is None:
                raise UserWarning("num_classes is None!")

        self.datamodule = datamodule
        self.num_classes = num_classes


class RayDataset(IterableDataset):
    def __init__(self, path: List[str], extensions: List[str] = None):
        self.path = path
        self.extensions = extensions
        self.ds = ray.data.read_binary_files(path, include_paths=True)

    def __iter__(self):
        return self.ds.iter_rows()

    def __len__(self):
        return len(self.input_files)

    def map_(self, func, *args, **kwargs) -> None:
        """Inplace Map for ray.data
        Time complexity: O(dataset size / parallelism)

        See https://docs.ray.io/en/latest/data/dataset.html#transforming-datasets"""
        self.ds = self.ds.map(func, *args, **kwargs)

    def map_batch_(self, func, batch_size: int = 2, *args, **kwargs) -> None:
        """Inplace Map for ray.data
        Time complexity: O(dataset size / parallelism)

        See https://docs.ray.io/en/latest/data/dataset.html#transforming-datasets"""
        self.ds = self.ds.map_batches(func, batch_size=batch_size, *args, **kwargs)

    @property
    def input_files(self):
        return self.ds.input_files()
