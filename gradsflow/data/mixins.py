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
from typing import Dict, List, Tuple, Union

import torch


class DataMixin:
    INPUT_KEY = 0  # other common value - inputs, images, text
    OUTPUT_KEY = 1  # other common values - target, ground

    def fetch_inputs(self, data: Union[List, Dict]):
        return data[self.INPUT_KEY]

    def fetch_target(self, data: Union[List, Dict]):
        return data[self.OUTPUT_KEY]

    @classmethod
    def send_to_device(cls, data: Union[List, Dict, Tuple, torch.Tensor, int, float]):
        """Send data to be device"""
        if isinstance(data, (int, float, str)):
            return data

        if isinstance(data, torch.Tensor):
            return data.to(cls.device)

        if isinstance(data, (list, tuple)):
            return list(map(cls.send_to_device, data))
        if isinstance(data, dict):
            return {k: cls.send_to_device(v) for k, v in data.items()}
        raise NotImplementedError(
            f"send_to_device is not implemented for data of type {type(data)}! Please raise an issue/pr"
        )
