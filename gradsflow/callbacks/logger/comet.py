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

from comet_ml import BaseExperiment

from gradsflow.callbacks import Callback
from gradsflow.utility.imports import requires

CURRENT_FILE = os.path.dirname(os.path.realpath(__file__))


class CometCallback(Callback):
    """
    [Comet](https://www.comet.ml/) Logging callback. To use this callback `pip install comet-ml`.
    Args:
        project_name: Name of the Project
        api_key: project API key
        offline: log experiment offline
    """

    def __init__(
        self,
        project_name: str = "awesome-project",
        api_key: Optional[str] = os.environ.get("COMET_API_KEY"),
        code_file: str = CURRENT_FILE,
        offline: bool = False,
        **kwargs
    ):
        super().__init__(
            model=None,
        )
        self._code_file = code_file
        self.experiment = self._create_experiment(project_name=project_name, api_key=api_key, offline=offline, **kwargs)

    @requires("comet_ml", "CometCallback requires comet_ml to be installed!")
    def _create_experiment(
        self, project_name: str, offline: bool = False, api_key: Optional[str] = None, **kwargs
    ) -> BaseExperiment:
        from comet_ml import Experiment, OfflineExperiment

        if offline:
            experiment = OfflineExperiment(project_name=project_name, **kwargs)
        else:
            experiment = Experiment(project_name=project_name, api_key=api_key, **kwargs)
        return experiment

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
        step = self.model.tracker.current_step
        outputs = kwargs["outputs"]
        loss = outputs["loss"].item()
        self.experiment.log_metrics(outputs.get("metrics", {}), step=step)
        self.experiment.log_metric("step_loss", loss, step=step)

    def on_val_step_end(self, *args, **kwargs):
        step = self.model.tracker.current_step
        outputs = kwargs["outputs"]
        loss = outputs["loss"].item()
        self.experiment.log_metrics(outputs.get("metrics", {}), step=step)
        self.experiment.log_metric("step_loss", loss, step=step)

    def on_epoch_end(self):
        step = self.model.tracker.current_step
        epoch = self.model.tracker.current_epoch
        train_loss = self.model.tracker.train_loss
        train_metrics = self.model.tracker.train_metrics
        val_loss = self.model.tracker.val_loss
        val_metrics = self.model.tracker.val_metrics

        self.experiment.train()
        self.experiment.log_metric("epoch_loss", train_loss, step=step, epoch=epoch)
        self.experiment.log_metrics(train_metrics, step=step, epoch=epoch)

        self.experiment.validate()
        self.experiment.log_metric("epoch_loss", val_loss, step=step, epoch=epoch)
        self.experiment.log_metrics(val_metrics, step=step, epoch=epoch)
        self.experiment.log_epoch_end(epoch)
