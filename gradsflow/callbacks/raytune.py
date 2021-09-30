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

from .callbacks import Callback

_METRICS = {
    "val_accuracy": "val_accuracy",
    "train_accuracy": "train_accuracy",
}


def report_checkpoint_callback(metrics: Optional[dict] = None, filename: Optional[str] = None):
    metrics = metrics or _METRICS
    filename = filename or "filename"
    callback = TuneReportCheckpointCallback(metrics=metrics, filename=filename, on="validation_end")

    return callback


class TorchTuneCheckpointCallback(Callback):
    def on_epoch_end(self):
        epoch = self.model.tracker.epoch
        model = self.model.learner

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "filename")
            torch.save((model.state_dict()), path)


class TorchTuneReport(Callback):
    def on_epoch_end(self):
        val_loss = self.model.tracker.val.loss
        train_loss = self.model.tracker.train.loss
        val_accuracy = self.model.tracker.tune_metric
        tune.report(val_loss=val_loss, val_accuracy=val_accuracy, train_loss=train_loss)
