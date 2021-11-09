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

from gradsflow.callbacks import Callback
from gradsflow.utility.common import to_item
from gradsflow.utility.imports import requires

os.environ["COMET_DISABLE_AUTO_LOGGING"] = "1"


class CometCallback(Callback):
    @requires("comet_ml", "pip install comet_ml to use CometCallback")
    def __init__(self, project_name: str = "awesome-project", api_key: Optional[str] = None):
        os.environ["COMET_DISABLE_AUTO_LOGGING"] = "1"
        from comet_ml import Experiment

        """
        Saves Model training metrics as CSV
        Args:
            project_name: Name of the Project
        """
        super().__init__(
            model=None,
        )
        self.experiment = Experiment(project_name=project_name, api_key=api_key)

    def on_train_epoch_start(
        self,
    ):
        self.experiment.train()

    def on_val_epoch_start(
        self,
    ):
        self.experiment.validate()

    def on_train_step_end(self, *args, **kwargs):
        outputs = kwargs["outputs"]
        loss = outputs["loss"].item()
        self.experiment.log_metrics(outputs.get("metrics", {}))
        self.experiment.log_metric("train_step_loss", loss)

    def on_val_step_end(self, *args, **kwargs):
        outputs = kwargs["outputs"]
        loss = outputs["loss"].item()
        self.experiment.log_metrics(outputs.get("metrics", {}))
        self.experiment.log_metric("val_step_loss", loss)

    def on_epoch_end(self):
        # TODO: cache this
        step = self.model.tracker.current_step
        epoch = self.model.tracker.current_epoch
        train_loss = self.model.tracker.train_loss
        val_loss = self.model.tracker.val_loss
        train_metrics = self.model.tracker.train_metrics
        val_metrics = self.model.tracker.val_metrics
        train_metrics = {"train/" + k: v.avg for k, v in train_metrics.items()}
        val_metrics = {"val/" + k: v.avg for k, v in val_metrics.items()}
        train_metrics = to_item(train_metrics)
        val_metrics = to_item(val_metrics)

        data = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **train_metrics, **val_metrics}
        print(data)
        self.experiment.log_metrics(data, step=step, epoch=epoch)
        self.experiment.log_epoch_end(epoch)
