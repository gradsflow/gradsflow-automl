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

from gradsflow import AutoDataset, Model
from gradsflow.data.image import get_fake_data

# Replace dataloaders with your custom dataset and you are all set to train your model
image_size = (64, 64)
num_classes = 2
train_dl = get_fake_data(image_size, num_classes=num_classes).dataloader
val_dl = get_fake_data(image_size, num_classes=num_classes).dataloader

autodataset = AutoDataset(train_dl, val_dl, num_classes=num_classes)


if __name__ == "__main__":
    cnn = create_model("resnet18", pretrained=False, num_classes=num_classes)

    model = Model(cnn)
    model.compile("crossentropyloss", "adam", metrics="accuracy")
    model.fit(autodataset, max_epochs=10, steps_per_epoch=50)
