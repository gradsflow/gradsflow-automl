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

image_size = (64, 64)
train_data = get_fake_data(image_size, num_classes=2)
val_data = get_fake_data(image_size, num_classes=2)

num_classes = train_data.dataset.num_classes
autodataset = AutoDataset(train_data.dataloader, val_data.dataloader, num_classes=num_classes)


def test_automodelv2():
    tuner = Tuner()
    cnn1 = create_model("resnet18", pretrained=False, num_classes=num_classes)

    cnns = tuner.suggest_complex("learner", cnn1)
    tuner.choice("optimizer", "sgd")
    tuner.scalar(
        "loss",
        "crossentropyloss",
    )
    model = AutoModelV2(cnns, optimization_metric="val_loss")
    model.hp_tune(
        tuner,
        autodataset,
        n_trials=1,
        epochs=1,
        cpu=1,
        trainer_config={"steps_per_epoch": 2},
    )
