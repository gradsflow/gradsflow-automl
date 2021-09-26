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
import os
from typing import Optional

import torch
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from gradsflow.core.base import BaseTracker

_METRICS = {
    "val_accuracy": "val_accuracy",
    "train_accuracy": "train_accuracy",
}


def report_checkpoint_callback(metrics: Optional[dict] = None, filename: Optional[str] = None):
    metrics = metrics or _METRICS
    filename = filename or "filename"
    callback = TuneReportCheckpointCallback(metrics=metrics, filename=filename, on="validation_end")

    return callback


class Callback:
    def __init__(self, tracker: BaseTracker = None):
        self.tracker = tracker

    def on_training_start(self):
        ...

    def on_epoch_start(
        self,
    ):
        ...

    def on_training_end(
        self,
    ):
        ...

    def on_epoch_end(self):
        ...


class TorchTuneCheckpointCallback(Callback):
    def on_epoch_end(self):
        epoch = self.tracker.epoch
        model = self.tracker.model
        optimizer = self.tracker.optimizer

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            print("checkpoint_dir", checkpoint_dir)
            path = os.path.join(checkpoint_dir, "filename")
            torch.save((model.state_dict(), optimizer.state_dict()), path)


class TorchTuneReport(Callback):
    def on_epoch_end(self):
        val_loss = self.tracker.val.loss
        train_loss = self.tracker.train.loss
        val_accuracy = self.tracker.tune_metric
        tune.report(loss=val_loss, val_accuracy=val_accuracy, train_loss=train_loss)


class ComposeCallback(Callback):
    _CALLBACK_INDEX = {
        "tune_checkpoint": TorchTuneCheckpointCallback,
        "tune_report": TorchTuneReport,
    }

    def available_callbacks(self):
        return list(self._CALLBACK_INDEX.keys())

    def __init__(self, tracker, *callbacks: str):
        super().__init__()
        self.callbacks = []
        for callback in callbacks:
            if isinstance(callback, str):
                callback = self._CALLBACK_INDEX[callback](tracker)
                self.callbacks.append(callback)
            elif isinstance(callback, Callback):
                self.callbacks.append(callback)
            else:
                raise NotImplementedError

    def on_epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end()

    def on_epoch_start(self):
        for callback in self.callbacks:
            callback.on_epoch_start()
