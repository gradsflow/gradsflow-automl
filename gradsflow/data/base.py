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
from typing import Dict, List, Union


class DataMixin:
    @staticmethod
    def fetch_inputs(data: Union[List, Dict]):
        return data[0]

    @staticmethod
    def fetch_target(data: Union[List, Dict]):
        return data[1]

    @staticmethod
    def send_to_device(batch: Union[List, Dict], device):
        """Send data to be device"""
        if isinstance(batch, (list, tuple)):
            return list(map(lambda x: x.to(device), batch))
        if isinstance(batch, dict):
            return {k: v.to(device) for k, v in batch.items()}
        raise NotImplementedError(
            f"send_to_device is not implemented for data of type {type(batch)}! Please raise an issue/pr"
        )
