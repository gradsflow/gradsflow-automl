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
from flash.image import ImageClassificationData

from gradsflow.core.automodel import AutoModel

datamodule = ImageClassificationData.from_folders(
    train_folder="data/hymenoptera_data/train/",
    val_folder="data/hymenoptera_data/val/",
)


def test_auto_model():
    assert AutoModel(datamodule)


def test_build_model():
    model = AutoModel(datamodule)
    with pytest.raises(NotImplementedError):
        model.build_model(**{"lr": 1})


def test_build_model():
    model = AutoModel(datamodule)
    with pytest.raises(NotImplementedError):
        model._objective(None)
