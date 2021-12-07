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
import inspect
import os
import re
import sys
import warnings
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import torch


def get_file_extension(path: str) -> str:
    """Returns extension of the file"""
    return os.path.basename(path).split(".")[-1]


def get_files(folder: str):
    """Fetch every file from given folder recursively."""
    folder = str(Path(folder) / "**" / "*")
    return glob(folder, recursive=True)


def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def module_to_cls_index(module, lower_key: bool = True) -> dict:
    """Fetch classes from module and create a Dictionary with key as class name and value as Class"""
    class_members = inspect.getmembers(sys.modules[module.__name__], inspect.isclass)
    mapping = {}
    for k, v in class_members:
        if lower_key:
            k = k.lower()
        mapping[k] = v

    return mapping


def listify(item: Any) -> List:
    """Convert any scalar value into list."""
    if not item:
        return []
    if isinstance(item, list):
        return item
    if isinstance(item, (tuple, set)):
        return list(item)
    if isinstance(item, (int, float, str)):
        return [item]
    try:
        return list(item)
    except TypeError:
        return [item]


# ref: https://github.com/rwightman/pytorch-image-models/blob/b544ad4d3fcd02057ab9f43b118290f2a089566f/timm/utils/metrics.py#L7
@dataclasses.dataclass(init=False)
class AverageMeter:
    """Computes and stores the average and current value.
    `val` is the running value, `avg` is the average value over an epoch.
    """

    name: Optional[str]
    avg: Optional[float] = 0

    def __init__(self, name=None):
        self.name = name
        self.computed = False
        self.val = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Updates the average meter value with new data. It also converts `torch.Tensor` to primitive datatype."""
        self.computed = True
        self.val = to_item(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def to_item(data: Union[torch.Tensor, Iterable, Dict]) -> Union[int, float, str, np.ndarray, Dict]:
    """
    Converts torch.Tensor into cpu numpy format.
    Args:
        data: torch.Tensor contained in any Iterable or Dictionary.
    """

    if isinstance(data, (int, float, str)):
        return data
    if isinstance(data, (list, tuple)):
        return type(data)(map(to_item, data))
    if isinstance(data, dict):
        return {k: to_item(v) for k, v in data.items()}

    if torch.is_tensor(data):
        if data.requires_grad:
            data = data.detach()
        data = data.cpu().numpy()

    warnings.warn("to_item didn't convert any value.")
    return data


def filter_list(arr: List[str], pattern: Optional[str] = None) -> List[str]:
    """Filter a list of strings with given pattern
    ```python
    >> arr = ['crossentropy', 'binarycrossentropy', 'softmax', 'mae',]
    >> filter_list(arr, ".*entropy*")
    >> # ["crossentropy", "binarycrossentropy"]
    ```
    """
    if pattern is None:
        return arr

    p = re.compile(pattern)
    return [s for s in arr if p.match(s)]
