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

from gradsflow import AutoDataset
from gradsflow.data.image import get_fake_data
from gradsflow.tuner import AutoModelV2, Tuner

image_size = (128, 128)
train_data = get_fake_data(image_size, num_workers=0)

val_data = get_fake_data(image_size, num_workers=0)

num_classes = train_data.dataset.num_classes
autodataset = AutoDataset(train_data.dataloader, val_data.dataloader, num_classes=num_classes)

cnn1 = create_model("resnet18", pretrained=False, num_classes=num_classes)
cnn2 = create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)

tuner = Tuner()

cnns = tuner.suggest_complex("learner", cnn1, cnn2)
optimizers = tuner.choice("optimizer", "adam", "sgd")
loss = tuner.choice(
    "loss",
    "crossentropyloss",
)


def test_automodelv2():
    model = AutoModelV2(cnns)
    model.hp_tune(tuner, autodataset, epochs=1)
