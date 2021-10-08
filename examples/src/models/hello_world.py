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
import torchvision
from timm import create_model
from torch.utils.data import DataLoader
from torchvision import transforms as T

from gradsflow import AutoDataset, Model
from gradsflow.data.common import random_split_dataset

# Replace dataloaders with your custom dataset and you are all set to train your model
image_size = (64, 64)
batch_size = 4

to_rgb = lambda x: x.convert("RGB")

augs = T.Compose([to_rgb, T.AutoAugment(), T.Resize(image_size), T.ToTensor()])
data = torchvision.datasets.Caltech101("~/", download=True, transform=augs)
train_data, val_data = random_split_dataset(data, 0.99)
train_dl = DataLoader(train_data, batch_size=batch_size)
val_dl = DataLoader(val_data, batch_size=batch_size)
num_classes = len(data.categories)

if __name__ == "__main__":
    autodataset = AutoDataset(train_dl, val_dl, num_classes=num_classes)
    cnn = create_model("resnet18", pretrained=False, num_classes=num_classes)

    model = Model(cnn)
    model.compile("crossentropyloss", "adam", metrics=["accuracy"])
    model.fit(autodataset, max_epochs=10, steps_per_epoch=50)
