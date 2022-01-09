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

from gradsflow.callbacks import CallbackRunner, TrainEvalCallback
from gradsflow.callbacks.base import Callback


def test_init(dummy_model):

    assert isinstance(CallbackRunner(dummy_model, "training").callbacks["TrainEvalCallback"], TrainEvalCallback)
    with pytest.raises(NotImplementedError):
        CallbackRunner(dummy_model, "random")


def test_append(dummy_model):

    cb = CallbackRunner(dummy_model)
    with pytest.raises(NotImplementedError):
        cb.append("random")
    cb.append("tune_checkpoint")
    cb.append(TrainEvalCallback(cb.model))
    assert len(cb.callbacks) == 2

    for cb_name, cb in cb.callbacks.items():
        assert isinstance(cb_name, str)
        assert isinstance(cb, Callback)


def test_clean(dummy_model):

    cb = CallbackRunner(dummy_model, TrainEvalCallback())
    cb.clean(keep="TrainEvalCallback")
    assert cb.callbacks.get("TrainEvalCallback") is not None
