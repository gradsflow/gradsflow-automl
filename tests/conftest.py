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
# Arrange
import pytest
import timm

from gradsflow import Model
from gradsflow.models.tracker import Tracker


@pytest.fixture
def resnet18():
    cnn = timm.create_model("ssl_resnet18", pretrained=False, num_classes=10).eval()

    return cnn


@pytest.fixture
def cnn_model(resnet18):
    model = Model(resnet18)
    model.TEST = True

    return model


@pytest.fixture
def tracker():
    return Tracker()
