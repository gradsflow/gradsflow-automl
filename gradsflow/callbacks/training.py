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

from gradsflow.core.callbacks import Callback


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
