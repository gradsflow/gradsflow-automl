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
from typing import Callable, Optional

import pytorch_lightning as pl
from accelerate import Accelerator
from loguru import logger
from torch.utils.data import DataLoader, Dataset

from gradsflow.core.data import BaseAutoDataset

from ..utility.common import default_device
from .mixins import DataMixin


class AutoDataset(BaseAutoDataset, DataMixin):
    _default_device = default_device()

    def __init__(
        self,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        datamodule: Optional[pl.LightningDataModule] = None,
        num_classes: Optional[int] = None,
    ):
        super().__init__()
        self.setup(train_dataloader, val_dataloader, train_dataset, val_dataset, datamodule, num_classes)

    def setup(
        self,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        datamodule: Optional[pl.LightningDataModule] = None,
        num_classes: Optional[int] = None,
    ):

        self.datamodule = datamodule
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_classes = num_classes

        if (datamodule or train_dataloader) is None:
            raise UserWarning("Both datamodule and _train_dataloader can't be None!")

        if all((datamodule, train_dataloader)):
            logger.warning("Both datamodule and _train_dataloader is set! Using datamodule over _train_dataloader.")

        if isinstance(datamodule, pl.LightningDataModule):
            self.datamodule = datamodule
            self._train_dataloader = datamodule.train_dataloader()
            self._val_dataloader = datamodule.val_dataloader()
            if hasattr(datamodule, "num_classes"):
                self.num_classes = datamodule.num_classes
            if hasattr(datamodule, "num_labels"):
                self.meta["num_labels"] = datamodule.num_labels

        self.meta["num_classes"] = self.num_classes

    @property
    def device_setup_status(self):
        return self.meta.get("device_setup_status")

    @device_setup_status.setter
    def device_setup_status(self, value: bool = True):
        logger.debug("setting device setup=True")
        self.meta["device_setup_status"] = value

    def prepare_data(self, accelerator: Accelerator) -> None:
        self._train_dataloader = accelerator.prepare_data_loader(self._train_dataloader)
        if self._val_dataloader:
            self._val_dataloader = accelerator.prepare_data_loader(self._val_dataloader)
        self.device_setup_status = True

    def fetch(self, data, device_mapper: Optional[Callable] = None):
        """
        If data is not sent to `device` then will attempt to map the `device_mapper` function on data.
        Args:
            data: Data Batch
            device_mapper: Function to move data to device
        """
        if self.device_setup_status:
            return data
        if device_mapper:
            data = map(device_mapper, data, self._default_device)
        return data

    @property
    def train_dataloader(self):
        return self.fetch(self._train_dataloader, self.send_to_device)

    @property
    def val_dataloader(self):
        return self.fetch(self._val_dataloader, self.send_to_device)
