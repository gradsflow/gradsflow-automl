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
from typing import Union

from gradsflow.callbacks.callbacks import Callback
from gradsflow.callbacks.progress import ProgressCallback
from gradsflow.callbacks.raytune import TorchTuneCheckpointCallback, TorchTuneReport

if typing.TYPE_CHECKING:
    from gradsflow.models.model import Model


class CallbackRunner(Callback):
    _AVAILABLE_CALLBACKS = {
        "tune_checkpoint": TorchTuneCheckpointCallback,
        "tune_report": TorchTuneReport,
        "progress": ProgressCallback,
    }

    def __init__(self, model: "Model", *callbacks: Union[str, Callback]):
        super().__init__(model)
        self.callbacks = []
        for callback in callbacks:
            if isinstance(callback, str):
                callback_fn = self._AVAILABLE_CALLBACKS[callback](model)
                self.callbacks.append(callback_fn)
            elif isinstance(callback, Callback):
                self.callbacks.append(callback)
            else:
                raise NotImplementedError

    def available_callbacks(self):
        return list(self._AVAILABLE_CALLBACKS.keys())

    def on_train_epoch_end(self):
        for callback in self.callbacks:
            callback.on_train_epoch_end()

    def on_train_epoch_start(self):
        for callback in self.callbacks:
            callback.on_train_epoch_start()

    def on_fit_start(self):
        for callback in self.callbacks:
            callback.on_fit_start()

    def on_fit_end(
        self,
    ):
        for callback in self.callbacks:
            callback.on_fit_end()

    def on_val_epoch_start(
        self,
    ):
        for callback in self.callbacks:
            callback.on_val_epoch_start()

    def on_val_epoch_end(self):
        for callback in self.callbacks:
            callback.on_val_epoch_end()

    def on_train_step_start(self):
        for callback in self.callbacks:
            callback.on_train_step_start()

    def on_train_step_end(self):
        for callback in self.callbacks:
            callback.on_train_step_end()

    def on_val_step_start(self):
        for callback in self.callbacks:
            callback.on_val_step_start()

    def on_val_step_end(self):
        for callback in self.callbacks:
            callback.on_val_step_end()

    def on_epoch_start(self):
        for callback in self.callbacks:
            callback.on_epoch_start()

    def on_epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end()
