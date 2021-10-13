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
import pytest
import timm
import torch

from gradsflow.data import AutoDataset
from gradsflow.data.image import get_fake_data
from gradsflow.models.model import Model
from gradsflow.models.tracker import Tracker

image_size = (64, 64)
train_data = get_fake_data(image_size)
val_data = get_fake_data(image_size)

num_classes = train_data.dataset.num_classes
autodataset = AutoDataset(train_data.dataloader, val_data.dataloader, num_classes=num_classes)

cnn = timm.create_model("ssl_resnet18", pretrained=False, num_classes=num_classes).eval()
model = Model(cnn)
model.compile("crossentropyloss", "adam")
model.TEST = True


def test_predict():
    x = torch.randn(1, 3, 64, 64)
    r1 = model.forward(x)
    r2 = model(x)
    r3 = model.predict(x)
    assert torch.all(torch.isclose(r1, r2))
    assert torch.all(torch.isclose(r2, r3))
    assert isinstance(model.predict(torch.randn(1, 3, 64, 64)), torch.Tensor)


def test_fit():
    model.TEST = True
    assert autodataset
    tracker = model.fit(autodataset, max_epochs=1, steps_per_epoch=1, show_progress=True)
    assert isinstance(tracker, Tracker)

    autodataset2 = AutoDataset(train_data.dataloader, num_classes=num_classes)
    model.TEST = False
    tracker2 = model.fit(autodataset2, max_epochs=1, steps_per_epoch=1, show_progress=False, resume=False)
    assert isinstance(tracker2, Tracker)


def test_compile():
    model1 = Model(cnn)

    def compute_accuracy(*_, **__):
        return 1

    with pytest.raises(NotImplementedError):
        model1.compile("crossentropyloss", "adam", metrics=compute_accuracy)

    with pytest.raises(AssertionError):
        model1.compile("crossentropyloss", "adam", metrics="random_val")

    model1.compile("crossentropyloss", "adam", metrics="accuracy")

    model2 = Model(cnn)
    model2.compile("crossentropyloss", torch.optim.Adam)
    model2.compile(torch.nn.CrossEntropyLoss, torch.optim.Adam, learning_rate=0.01)
    assert model2.optimizer.param_groups[0]["lr"] == 0.01


def test_set_accelerator():
    model2 = Model(cnn, accelerator_config={"fp16": True})
    model2.compile()
    assert model2.accelerator
