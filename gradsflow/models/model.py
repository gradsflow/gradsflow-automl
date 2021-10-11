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
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from loguru import logger
from torch import nn
from torchmetrics import Metric

from gradsflow.callbacks import Callback, CallbackRunner
from gradsflow.callbacks.progress import ProgressCallback
from gradsflow.callbacks.training import TrainEvalCallback
from gradsflow.data import AutoDataset
from gradsflow.data.mixins import DataMixin
from gradsflow.models.base import BaseModel
from gradsflow.models.exceptions import EpochCancel, FitCancel
from gradsflow.models.tracker import Tracker
from gradsflow.utility.common import listify, module_to_cls_index

METRICS_TYPE = Union[str, Metric, List[Union[str, Metric]], None]


class Model(BaseModel, DataMixin):
    """
    Model provide training functionality with `model.fit(...)` inspired from Keras

    Examples:
    ```python
    model = Model(cnn)
    model.compile("crossentropyloss", "adam", learning_rate=1e-3, metrics="accuracy")
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
        device: Optional[str] = None,
        use_accelerate: bool = True,
        accelerator_config: dict = None,
    ):
        accelerator_config = accelerator_config or {}
        super().__init__(
            learner=learner,
            device=device,
            use_accelerate=use_accelerate,
            accelerator_config=accelerator_config,
        )
        self.callback_runner: Optional[CallbackRunner] = None

    def forward_once(self, x) -> torch.Tensor:
        self.callback_runner.on_forward_start()
        x = self.forward(x)
        self.callback_runner.on_forward_end()
        return x

    def compile(
        self,
        loss: Union[str, nn.modules.loss._Loss] = None,
        optimizer: Union[str, Callable] = None,
        learning_rate: float = 3e-4,
        metrics: METRICS_TYPE = None,
        loss_config: Optional[dict] = None,
        optimizer_config: Optional[dict] = None,
    ) -> None:
        """
        Examples:
            ```python
            model = Model(net)
            model.compile(loss="crossentropyloss", optimizer="adam", learning_rate=1e-3, metrics="accuracy")
            ```
        Args:
            loss: name of loss or torch Loss class object. See `available_losses()`
            optimizer: optimizer name or `torch.optim.Optimizer` Class
            learning_rate: defaults to 1e-3
            metrics: list of metrics to calculate. See `available_metrics()`
            loss_config: Dict config if any to pass to loss function
            optimizer_config: Dict config if any to pass to Optimizer

        """
        loss_config = loss_config or {}
        optimizer_config = optimizer_config or {}

        if optimizer:
            optimizer_fn = self._get_optimizer(optimizer)
            optimizer = optimizer_fn(self.learner.parameters(), lr=learning_rate, **optimizer_config)
            self.optimizer = self.prepare_optimizer(optimizer)
        if loss:
            self.loss = self._get_loss(loss)(**loss_config)
        self.add_metrics(*listify(metrics))
        self._compiled = True

    def calculate_metrics(self, preds, target) -> Dict[str, torch.Tensor]:
        self.metrics.update(preds, target)
        return self.metrics.compute()

    def train_step(self, batch: Union[List[torch.Tensor], Dict[Any, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        inputs = self.fetch_inputs(batch)
        target = self.fetch_target(batch)

        self.optimizer.zero_grad()
        logits = self.forward_once(inputs)
        loss = self.loss(logits, target)
        self.backward(loss)
        self.optimizer.step()

        self.tracker.track("train/step_loss", loss, render=True)
        return {"loss": loss, "metrics": self.calculate_metrics(logits, target)}

    def val_step(self, batch: Union[List[torch.Tensor], Dict[Any, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        inputs = self.fetch_inputs(batch)
        target = self.fetch_target(batch)

        logits = self.forward_once(inputs)
        loss = self.loss(logits, target)

        self.tracker.track("val/step_loss", loss, render=True)
        return {"loss": loss, "metrics": self.calculate_metrics(logits, target)}

    def train_one_epoch(self, train_dataloader):
        tracker = self.tracker
        tracker.train.steps = 0
        steps_per_epoch = tracker.steps_per_epoch

        for step, batch in enumerate(train_dataloader):
            tracker.train.steps = step
            # ----- TRAIN STEP -----
            self.callback_runner.on_train_step_start()
            outputs = self.train_step(batch)
            self.callback_runner.on_train_step_end(outputs=outputs)
            if self.TEST:
                break
            if steps_per_epoch and step >= steps_per_epoch:
                break

    def val_one_epoch(self, val_dataloader):
        tracker = self.tracker
        tracker.val.steps = 0
        for step, batch in enumerate(val_dataloader):
            tracker.val.steps = step
            # ----- VAL STEP -----
            self.callback_runner.on_val_step_start()
            outputs = self.val_step(batch)
            self.callback_runner.on_val_step_end(outputs=outputs)
            if self.TEST:
                break

    def _train_epoch_with_event(self):
        train_dataloader = self.tracker.autodataset.get_train_dl(self.send_to_device)
        # ----- TRAIN -----
        self.callback_runner.on_train_epoch_start()
        self.train_one_epoch(train_dataloader)
        self.callback_runner.on_train_epoch_end()

    def _val_epoch_with_event(self):
        autodataset = self.tracker.autodataset
        if not autodataset.val_dataloader:
            return
        val_dataloader = self.tracker.autodataset.get_val_dl(self.send_to_device)
        # ------ VALIDATE -----
        self.callback_runner.on_val_epoch_start()
        self.val_one_epoch(val_dataloader)
        self.callback_runner.on_val_epoch_end()

    def epoch(self):
        current_epoch, max_epochs = self.tracker.current_epoch, self.tracker.max_epochs

        for epoch in range(current_epoch, max_epochs):
            self.tracker.current_epoch = epoch
            # ----- EPOCH -----
            self.callback_runner.on_epoch_start()
            self._train_epoch_with_event()
            self._val_epoch_with_event()
            self.callback_runner.on_epoch_end()

            if self.TEST:
                break

    def _fit_with_event(self):
        self.callback_runner.on_fit_start()
        self.callback_runner.with_event("epoch", self.epoch, EpochCancel)
        self.callback_runner.on_fit_end()

    def fit(
        self,
        autodataset: AutoDataset,
        max_epochs: int = 1,
        steps_per_epoch: Optional[int] = None,
        callbacks: Union[List[str], Callback] = None,
        resume: bool = True,
        show_progress: bool = True,
        progress_kwargs=None,
    ) -> Tracker:
        """
        Similar to Keras model.fit(...) it trains the model for specified epochs and returns Tracker object

        Examples:
            ```python
            autodataset = AutoDataset(train_dataloader, val_dataloader)
            model = Model(cnn)
            model.compile("crossentropyloss", "adam", learning_rate=1e-3, metrics="accuracy")
            model.fit(autodataset)
            ```

        Args:
            autodataset: AutoDataset object encapsulate dataloader and datamodule
            max_epochs: number of epochs to train
            steps_per_epoch: Number of steps trained in a single current_epoch
            callbacks: Callback object or string
            resume: Resume training from the last current_epoch
            show_progress: Enable to show training progress
            progress_kwargs: Arguments for rich.progress

        Returns:
            Tracker object
        """
        if not resume:
            self.tracker.reset()
        self.assert_compiled()
        callback_list = listify(callbacks)
        callback_list.append(TrainEvalCallback(self))
        if show_progress:
            callback_list.append(ProgressCallback(self, progress_kwargs))
        self.callback_runner = CallbackRunner(self, *callback_list)
        self.tracker.autodataset = self.prepare_data(autodataset)
        self.tracker.steps_per_epoch = steps_per_epoch
        self.tracker.max_epochs = max_epochs

        try:
            self.callback_runner.with_event("fit", self._fit_with_event, FitCancel)
        except KeyboardInterrupt:
            logger.error("Keyboard interruption detected")
        finally:
            self.callback_runner.clean()

        return self.tracker
