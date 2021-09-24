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
import io
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from gradsflow.data.ray_dataset import RayDataset


class RayImageFolder(RayDataset):
    """Read image datasets
    ```
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png
    ```
    """

    def __init__(self, path, transform: Union[Callable, None] = None):
        super().__init__(path)
        self.transform = transform

    @staticmethod
    def file_to_class(files: Union[str, List]):
        file_list = []
        if isinstance(files, (tuple, list)):
            for file in files:
                file_list.append(file.split("/")[-2])
            return file_list
        return files.split("/")[-2]

    def find_classes(self) -> List[str]:
        files = self.input_files
        return sorted(list(set(map(self.file_to_class, files))))

    def __iter__(self):
        for data in self.ds.iter_rows():
            x = Image.open(io.BytesIO(data[1]))
            target = self.file_to_class(data[0])
            if self.transform:
                x = self.transform(x)
            yield x, target


def get_augmentations(image_size: tuple = (224, 224), auto_augment_policy: bool = True):
    if auto_augment_policy:
        augmentations = [T.Resize(image_size), T.AutoAugment(), T.ToTensor()]
    else:
        augmentations = [T.Resize(image_size), T.ToTensor()]
    return T.Compose(augmentations)


def image_dataset_from_directory(
    directory: Union[List[str], Path, str],
    transform=None,
    image_size=(224, 224),
    batch_size: int = 1,
    shuffle: bool = False,
    pin_memory: bool = True,
    num_workers: Optional[int] = None,
    ray_data: bool = False,
) -> Dict[str, Union[RayDataset, DataLoader]]:
    """
    Create Dataset and Dataloader for image folder dataset.
    Args:
        directory:
        transform:
        image_size:
        batch_size:
        shuffle:
        pin_memory:
        num_workers:

    Returns:
        A dictionary containing dataset and dataloader.
    """

    num_workers = num_workers or os.cpu_count()
    if transform is True:
        transform = get_augmentations(image_size)
    if ray_data:
        ds = RayImageFolder(directory, transform=transform)
    else:
        ds = ImageFolder(directory, transform=transform)
    logger.info("ds created")
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        pin_memory=pin_memory,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return {"ds": ds, "dl": dl}
