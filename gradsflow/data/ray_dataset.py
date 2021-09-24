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
from typing import List, Union

import ray
from torch.utils.data import IterableDataset


class RayDataset(IterableDataset):
    def __init__(self, path: Union[List[str], str]):
        self.path = path
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

    def map_batch_(self, func, batch_size: int = 2, **kwargs) -> None:
        """Inplace Map for ray.data
        Time complexity: O(dataset size / parallelism)
        See https://docs.ray.io/en/latest/data/dataset.html#transforming-datasets"""
        self.ds = self.ds.map_batches(func, batch_size=batch_size, **kwargs)

    @property
    def input_files(self):
        return self.ds.input_files()
