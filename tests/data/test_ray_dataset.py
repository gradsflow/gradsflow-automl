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
from pathlib import Path

import numpy as np
from PIL import Image

from gradsflow.data.ray_dataset import RayDataset, RayImageFolder

data_dir = Path.cwd()


# TODO: remote dataset test
def test_ray_dataset():
    folder = f"{data_dir}/data/test-data-cat-dog-v0/cat-dog/"

    dataset = RayDataset(folder)

    assert len(dataset) == 8
    assert next(iter(dataset))

    assert dataset


def test_ray_image_folder():
    folder = f"{data_dir}/data/test-data-cat-dog-v0/cat-dog/"

    dataset = RayImageFolder(folder)

    # test_find_classes
    assert dataset.find_classes() == ["cat", "dog"]

    # test_iter
    item = next(iter(dataset))
    assert isinstance(item[0], Image.Image)
    assert isinstance(item[1], str)
