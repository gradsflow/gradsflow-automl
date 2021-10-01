"""Data loader for image dataset"""
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
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from gradsflow.core.data import Data
from gradsflow.data.ray_dataset import RayDataset, RayImageFolder

logger = logging.getLogger("data.image")


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


def get_fake_data(
    image_size: Tuple[int, int], num_classes=10, batch_size=1, pin_memory=False, shuffle=True, num_workers=0
):
    from torchvision.datasets import FakeData

    data = Data()

    transform = get_augmentations(
        image_size=image_size,
    )
    data.dataset = FakeData(size=100, image_size=[3, *image_size], num_classes=num_classes, transform=transform)
    data.dataloader = DataLoader(
        data.dataset,
        batch_size=batch_size,
        pin_memory=pin_memory,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return data
