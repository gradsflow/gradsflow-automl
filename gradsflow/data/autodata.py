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

from gradsflow.core.data import BaseAutoDataset

from .mixins import DataMixin


class AutoDataset(BaseAutoDataset, DataMixin):
    @property
    def sent_to_device(self):
        return self.meta.get("sent_to_device")

    @sent_to_device.setter
    def sent_to_device(self, value: bool = True):
        self.meta["sent_to_device"] = value

    def fetch(self, data, device_mapper: Optional[Callable] = None):
        """
        If data is not sent to `device` then will attempt to map the `device_mapper` function on data.
        Args:
            data: Data Batch
            device_mapper: Function to move data to device
        """
        if self.sent_to_device:
            return data
        if device_mapper:
            data = map(device_mapper, data)
        return data

    def get_train_dl(self, mapper_fn: Optional[Callable]):
        return self.fetch(self.train_dataloader, mapper_fn)

    def get_val_dl(self, mapper_fn: Optional[Callable]):
        return self.fetch(self.val_dataloader, mapper_fn)
