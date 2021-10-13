"""Provide some common functionalities/utilities for Datasets"""
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
from typing import List

from torch.utils.data import Dataset, random_split


def random_split_dataset(data: Dataset, pct=0.9) -> List[Dataset]:
    """
    Randomly splits dataset into two sets. Length of first split is len(data) * pct.
    Args:
        data: pytorch Dataset object with `__len__` implementation.
        pct: percentage of split.
    """
    n = len(data)
    split_1 = int(n * pct)
    split_2 = n - split_1
    return random_split(data, (split_1, split_2))
