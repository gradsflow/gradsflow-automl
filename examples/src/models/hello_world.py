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
from timm import create_model
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData

from gradsflow import AutoDataset, Model
from gradsflow.data.image import get_augmentations

image_size = (128, 128)
transform = get_augmentations(image_size)
train_ds = FakeData(size=100, image_size=[3, *image_size], transform=transform)
val_ds = FakeData(size=100, image_size=[3, *image_size], transform=transform)
train_dl = DataLoader(train_ds)
val_dl = DataLoader(val_ds)

num_classes = train_ds.num_classes
autodataset = AutoDataset(train_dl, val_dl, num_classes=num_classes)

cnn = create_model("resnet18", pretrained=False, num_classes=num_classes)

model = Model(cnn)
model.compile("crossentropyloss", "adam")
model.fit(autodataset, epochs=10)
