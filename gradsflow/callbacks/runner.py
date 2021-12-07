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
import typing
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

from gradsflow.callbacks.progress import ProgressCallback
from gradsflow.callbacks.raytune import TorchTuneCheckpointCallback, TorchTuneReport
from gradsflow.callbacks.training import TrainEvalCallback
from gradsflow.core.callbacks import Callback
from gradsflow.utility import listify

if typing.TYPE_CHECKING:
    from gradsflow.models.model import Model


class CallbackRunner(Callback):
    _name: str = "CallbackRunner"
    _AVAILABLE_CALLBACKS: Dict[str, Any] = {
        "training": TrainEvalCallback,
        "tune_checkpoint": TorchTuneCheckpointCallback,
        "tune_report": TorchTuneReport,
        "progress": ProgressCallback,
    }

    def __init__(self, model: "Model", *callbacks: Union[str, Callback]):
        super().__init__(model)
        self.callbacks = []
        self.callbacks = OrderedDict()
        for callback in callbacks:
            self.append(callback)

    # skipcq: W0212
    def append(self, callback: Union[str, Callback]):
        try:
            if isinstance(callback, str):
                callback_fn: Callback = self._AVAILABLE_CALLBACKS[callback](model=self.model)
                self.callbacks[callback_fn._name] = callback_fn
            elif isinstance(callback, Callback):
                callback.model = self.model
                self.callbacks[callback._name] = callback
        except KeyError:
            raise NotImplementedError(f"callback is not implemented {callback}")

    def available_callbacks(self):
        return list(self._AVAILABLE_CALLBACKS.keys())

    def on_train_epoch_end(self, *args, **kwargs):
        for _, callback in self.callbacks.items():
            callback.on_train_epoch_end(*args, **kwargs)

    def on_train_epoch_start(self):
        for _, callback in self.callbacks.items():
            callback.on_train_epoch_start()

    def on_fit_start(self):
        for _, callback in self.callbacks.items():
            callback.on_fit_start()

    def on_fit_end(
        self,
    ):
        for _, callback in self.callbacks.items():
            callback.on_fit_end()

    def on_val_epoch_start(
        self,
    ):
        for _, callback in self.callbacks.items():
            callback.on_val_epoch_start()

    def on_val_epoch_end(self, *args, **kwargs):
        for _, callback in self.callbacks.items():
            callback.on_val_epoch_end(*args, **kwargs)

    def on_train_step_start(self):
        for _, callback in self.callbacks.items():
            callback.on_train_step_start()

    def on_train_step_end(self, *args, **kwargs):
        for _, callback in self.callbacks.items():
            callback.on_train_step_end(*args, **kwargs)

    def on_val_step_start(self):
        for _, callback in self.callbacks.items():
            callback.on_val_step_start()

    def on_val_step_end(self, *args, **kwargs):
        for _, callback in self.callbacks.items():
            callback.on_val_step_end(*args, **kwargs)

    def on_epoch_start(self):
        for _, callback in self.callbacks.items():
            callback.on_epoch_start()

    def on_epoch_end(self):
        for _, callback in self.callbacks.items():
            callback.on_epoch_end()

    def on_forward_start(self):
        for _, callback in self.callbacks.items():
            callback.on_forward_start()

    def on_forward_end(self):
        for _, callback in self.callbacks.items():
            callback.on_forward_end()

    def clean(self, keep: Optional[Union[List[str], str]] = None):
        """Remove all the callbacks except callback names provided in keep"""
        for _, callback in self.callbacks.items():
            callback.clean()
        not_keep = set(self.callbacks.keys()) - set(listify(keep))
        for key in not_keep:
            self.callbacks.pop(key)
        # self.callbacks = OrderedDict(list(self.callbacks.items())[0:1])
