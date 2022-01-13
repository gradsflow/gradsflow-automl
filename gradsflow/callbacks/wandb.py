#  Copyright (c) 2022 GradsFlow. All rights reserved.
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

import wandb

from gradsflow.callbacks.base import Callback
from gradsflow.utility.imports import requires

CURRENT_FILE = os.path.dirname(os.path.realpath(__file__))


class WandbCallback(Callback):
    """
    [Weights & Biases](https://www.wandb.com/) Logging callback. To use this callback `pip install wandb`.
    Args:
        log_model: Whether to upload model artifact to Wandb
        code_file: path of the code you want to upload as artifact to Wandb
    """

    @requires("wandb", "WandbCallback requires wandb to be installed!")
    def __init__(
        self,
        log_model: bool = False,
        code_file: Optional[str] = None,
    ):
        super().__init__()
        if wandb.run is None:
            raise ValueError("You must call wandb.init() before WandbCallback()")
        self._code_file = code_file
        self._train_prefix = "train"
        self._val_prefix = "val"
        self._log_model = log_model

    def on_fit_start(self):
        if self._log_model:
            wandb.log_artifact(self.model.learner)
        if self._code_file:
            wandb.log_artifact(self._code_file)

    def _apply_prefix(self, data: dict, prefix: str):
        data = {f"{prefix}_{k}": v for k, v in data.items()}
        return data

    def _step(self, prefix: str, outputs: dict):
        step = self.model.tracker.mode(prefix).steps
        loss = outputs["loss"].item()
        metrics = outputs.get("metrics", {})
        metrics = self._apply_prefix(metrics, prefix)
        wandb.log(metrics, step=step)
        wandb.log({f"{prefix}_step_loss": loss}, step=step)

    def on_train_step_end(self, outputs: dict = None, **_):
        self._step(prefix=self._train_prefix, outputs=outputs)

    def on_val_step_end(self, outputs: dict = None, **_):
        self._step(prefix=self._val_prefix, outputs=outputs)

    def on_epoch_end(self):
        epoch = self.model.tracker.current_epoch
        train_loss = self.model.tracker.train_loss
        train_metrics = self.model.tracker.train_metrics
        val_loss = self.model.tracker.val_loss
        val_metrics = self.model.tracker.val_metrics

        train_metrics = self._apply_prefix(train_metrics, prefix=self._train_prefix)
        val_metrics = self._apply_prefix(val_metrics, prefix=self._val_prefix)

        wandb.log({"train_epoch_loss": train_loss, "epoch": epoch})
        wandb.log({"val_epoch_loss": val_loss, "epoch": epoch})
        wandb.log(train_metrics.update({"epoch": epoch}))
        wandb.log(val_metrics.update({"epoch": epoch}))
