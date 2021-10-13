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

from gradsflow.callbacks import Callback, CallbackRunner, TrainEvalCallback


def test_init():
    class DummyModel:
        def forward(self):
            return 1

    assert isinstance(CallbackRunner(DummyModel(), "training").callbacks[0], TrainEvalCallback)
    with pytest.raises(NotImplementedError):
        CallbackRunner(DummyModel(), "random")


def test_append():
    class DummyModel:
        def forward(self):
            return 1

    cb = CallbackRunner(DummyModel())
    with pytest.raises(NotImplementedError):
        cb.append("random")
    cb.append("training")
    cb.append(TrainEvalCallback(cb.model))
    assert len(cb.callbacks) == 2

    for e in cb.callbacks:
        assert isinstance(e, Callback)
