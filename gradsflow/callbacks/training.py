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

from gradsflow.callbacks.base import Callback


class TrainEvalCallback(Callback):
    _name = "TrainEvalCallback"

    def on_train_step_start(self):
        self.model.optimizer.zero_grad()

    def on_train_step_end(self, *args, **kwargs):
        # ----- AUTO OPTIMIZATION -----
        outputs = kwargs["outputs"]
        if not self.model.disable_auto_optimization:
            self.model.backward(outputs["loss"])
            self.model.optimizer.step()

        # ----- METRIC UPDATES -----
        tracker = self.model.tracker
        loss = outputs["loss"].item()
        tracker.val.step_loss = loss
        tracker.track_loss(loss, mode="train")
        tracker.track_metrics(outputs.get("metrics", {}), mode="train")

    def on_val_step_end(self, *args, **kwargs):
        # ----- METRIC UPDATES -----
        tracker = self.model.tracker
        outputs = kwargs["outputs"]
        loss = outputs["loss"].item()
        tracker.val.step_loss = loss
        tracker.track_loss(loss, mode="val")
        tracker.track_metrics(outputs.get("metrics", {}), mode="val")

    def on_train_epoch_start(self):
        self.model.train()
        self.model.metrics.reset()
        self.model.tracker.train.reset_metrics()

    def on_val_epoch_start(self):
        self.model.eval()
        self.model.metrics.reset()
        self.model.tracker.val.reset_metrics()


class EarlyStopping(Callback):
    # Port of Keras EarlyStopping Callback
    # https://github.com/keras-team/keras/blob/f5fea878c271e38946c6681c1c2434e72d0ab977/keras/callbacks.py#L1744
    # Thanks to Keras Team and contributors

    def __init__(
        self,
        monitor="val_loss",
        min_delta=0,
        patience=0,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=False,
    ):
        super().__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights
        # Tracking values
        self.best_model = None
        self.wait = 0
        self.best_epoch = 0
        self.stopped_epoch = 0

    def get_monitor_value(self, logs):
        pass

    def on_fit_start(self):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_model = None
        self.best_epoch = 0

    def on_epoch_end(self, logs: dict = None):
        pass

    # Restore the weights after first epoch if no progress is ever made.
    # Only reset wait if we beat both the baseline and our previous best.
    # Only check after the first epoch.
