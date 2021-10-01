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
from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
from torchmetrics import Metric

from gradsflow.callbacks import Callback, CallbackRunner
from gradsflow.callbacks.progress import ProgressCallback
from gradsflow.core.data import AutoDataset
from gradsflow.models.base import BaseModel
from gradsflow.models.tracker import Tracker
from gradsflow.utility.common import listify, module_to_cls_index


class Model(BaseModel):
    """
    Model provide training functionality with `model.fit(...)` inspired from Keras

    Examples:
    ```python
        model = Model(cnn)
        model.compile("crossentropyloss", "adam", learning_rate=1e-3)
        model.fit(autodataset)
    ```

    Args:
        learner: Trainable model
        accelerator_config: HuggingFace Accelerator config
    """

    TEST = os.environ.get("GF_CI", "false").lower() == "true"
    _OPTIMIZER_INDEX = module_to_cls_index(torch.optim, True)

    def __init__(
        self,
        learner: Union[nn.Module, Any],
        accelerator_config: dict = None,
    ):
        accelerator_config = accelerator_config or {}
        super().__init__(learner=learner, accelerator_config=accelerator_config)
        self.tracker = Tracker()

    def compile(
        self,
        loss=None,
        optimizer="adam",
        learning_rate=3e-4,
        metrics: Union[str, Metric, None] = None,
        loss_config: Optional[dict] = None,
        optimizer_config: Optional[dict] = None,
    ) -> None:
        loss_config = loss_config or {}
        optimizer_config = optimizer_config or {}
        optimizer_fn = self._get_optimizer(optimizer)
        optimizer = optimizer_fn(self.learner.parameters(), lr=learning_rate, **optimizer_config)
        self.loss = self._get_loss(loss)(**loss_config)
        self.add_metrics(*listify(metrics))
        self.prepare_optimizer(optimizer)
        self._compiled = True

    def train_step(self, inputs: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.optimizer.zero_grad()
        logits = self.learner(inputs)
        loss = self.loss(logits, target)
        self.accelerator.backward(loss)
        self.optimizer.step()
        return {"loss": loss, "logits": logits}

    def val_step(self, inputs: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.learner(inputs)
        loss = self.loss(logits, target)
        _, predictions = torch.max(logits.data, 1)
        return {"loss": loss, "logits": logits, "predictions": predictions}

    def train_one_epoch(self):
        self.tracker.callback_runner.on_train_epoch_start()
        train_dataloader = self.tracker.autodataset.train_dataloader
        tracker = self.tracker
        running_train_loss = 0.0
        tracker.train.steps = 0
        steps_per_epoch = tracker.steps_per_epoch

        self.learner.train()
        for step, (inputs, target) in enumerate(train_dataloader):
            self.tracker.callback_runner.on_train_step_start()
            outputs = self.train_step(inputs, target)
            self.tracker.callback_runner.on_train_step_end()
            loss = outputs["loss"].item()
            self.metrics.update(outputs["logits"], target)
            self.tracker.train.metrics = self.metrics.compute()

            running_train_loss += loss
            tracker.train.steps += 1
            if self.TEST:
                break
            if steps_per_epoch and step >= steps_per_epoch:
                break
        tracker.train.loss = running_train_loss / (tracker.train.steps + 1e-9)
        self.tracker.callback_runner.on_train_epoch_end()
        self.metrics.reset()

    def val_one_epoch(self):
        self.tracker.callback_runner.on_val_epoch_start()
        autodataset = self.tracker.autodataset
        if not autodataset.val_dataloader:
            return
        val_dataloader = autodataset.val_dataloader
        tracker = self.tracker
        tracker.total = 0
        tracker.correct = 0
        running_val_loss = 0.0
        tracker.val.steps = 0

        self.learner.eval()
        for _, (inputs, target) in enumerate(val_dataloader):
            with torch.no_grad():
                self.tracker.callback_runner.on_val_step_start()
                outputs = self.val_step(inputs, target)
                self.tracker.callback_runner.on_val_step_end()
                loss = outputs["loss"]
                predicted = outputs["predictions"]
                self.metrics.update(outputs["logits"], target)
                self.tracker.val.metrics = self.metrics.compute()

                tracker.total += target.size(0)
                tracker.correct += (predicted == target).sum().item()
                running_val_loss += loss.cpu().numpy()
                tracker.val.steps += 1
            if self.TEST:
                break
        tracker.val.loss = running_val_loss / (tracker.val.steps + 1e-9)
        tracker.tune_metric = tracker.val_accuracy = tracker.correct / tracker.val.steps
        self.tracker.callback_runner.on_val_epoch_end()
        self.metrics.reset()

    def fit(
        self,
        autodataset: AutoDataset,
        max_epochs: int = 1,
        steps_per_epoch: Optional[int] = None,
        callbacks: Union[List[str], Callback] = None,
        resume: bool = True,
        progress_kwargs=None,
    ) -> Tracker:
        """
        Similar to Keras model.fit(...) it trains the model for specified epochs and returns Tracker object
        Args:
            autodataset: AutoDataset object encapsulate dataloader and datamodule
            max_epochs: number of epochs to train
            steps_per_epoch: Number of steps trained in a single epoch
            callbacks: Callback object or string
            resume: Resume training from the last epoch
            progress_kwargs: Arguments for rich.progress

        Returns:
            Tracker object
        """
        progress_kwargs = progress_kwargs or {}
        if not resume:
            self.tracker.reset()
        self.assert_compiled()
        callback_list = listify(callbacks) + [ProgressCallback(self, **progress_kwargs)]
        self.tracker.callback_runner = CallbackRunner(self, *callback_list)
        self.tracker.autodataset = self.prepare_data(autodataset)
        self.tracker.steps_per_epoch = steps_per_epoch

        tracker = self.tracker

        # ----- EVENT: ON_TRAINING_START
        tracker.callback_runner.on_fit_start()

        for epoch in range(tracker.epoch, max_epochs):
            tracker.epoch = epoch

            # ----- EVENT: ON_EPOCH_START
            tracker.callback_runner.on_epoch_start()

            self.train_one_epoch()

            # END OF TRAIN EPOCH
            self.val_one_epoch()

            # ----- EVENT: ON_EPOCH_END
            tracker.callback_runner.on_epoch_end()

            if self.TEST:
                break

        # ----- EVENT: ON_TRAINING_END
        tracker.callback_runner.on_fit_end()

        return tracker
