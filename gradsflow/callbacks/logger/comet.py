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

CURRENT_FILE = os.path.dirname(os.path.realpath(__file__))


class CometCallback(Callback):
    """
    [Comet](https://www.comet.ml/) Logging callback. To use this callback `pip install comet-ml`.
    Args:
        project_name: Name of the Project
        api_key: project API key
    """

    @requires("comet_ml", "CometCallback requires comet_ml to be installed!")
    def __init__(
        self,
        project_name: str = "awesome-project",
        api_key: Optional[str] = os.environ.get("COMET_API_KEY"),
        code_file: str = CURRENT_FILE,
    ):
        from comet_ml import Experiment

        super().__init__(
            model=None,
        )
        self._code_file = code_file
        self.experiment = Experiment(project_name=project_name, api_key=api_key)

    def on_fit_start(self):
        self.experiment.set_model_graph(self.model.learner)
        self.experiment.set_code(self._code_file)

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
        self.experiment.log_metrics(data, step=step, epoch=epoch)
        self.experiment.log_epoch_end(epoch)
