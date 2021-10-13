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
import pytest
import torch

from gradsflow.data.mixins import DataMixin
from gradsflow.utility import default_device


class DataTest(DataMixin):
    device = default_device()


datamixin = DataTest()


def test_send_to_device():
    # data as primitive
    assert datamixin.send_to_device(1) == 1
    assert datamixin.send_to_device(1.5) == 1.5

    # data as Tensor
    x = torch.randn(4, 1)
    assert isinstance(datamixin.send_to_device(x), torch.Tensor)

    # data as list
    batch = torch.randn(4, 16), [1] * 4
    assert datamixin.send_to_device(batch)

    # data as dict
    batch = {"inputs": torch.randn(4, 16), "targets": [1] * 4}
    assert datamixin.send_to_device(batch)

    # catch error
    with pytest.raises(NotImplementedError):
        datamixin.send_to_device(set(batch))
