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
from typing import Dict, List, Optional, Union

import torch
from rich.progress import BarColumn, Progress, RenderableColumn, TimeRemainingColumn
from torch import nn

from gradsflow.core.callbacks import ComposeCallback
from gradsflow.core.data import AutoDataset
from gradsflow.model.base import BaseModel
from gradsflow.model.tracker import Tracker
from gradsflow.utility.common import listify, module_to_cls_index


class Model(BaseModel):
    TEST = os.environ.get("GF_CI", "false").lower() == "true"
    _OPTIMIZER_INDEX = module_to_cls_index(torch.optim, True)

    def __init__(self, model: nn.Module, optimizer: str, lr: float = 3e-4, device=None):
        optimizer = self._OPTIMIZER_INDEX[optimizer](model.parameters(), lr=lr)
        super().__init__(model=model, optimizer=optimizer, lr=lr, device=device)

        self.criterion = nn.CrossEntropyLoss()
        self.tracker = Tracker()
        self.tracker.model = model

    def train_step(self, inputs: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        inputs, target = inputs.to(self.device), target.to(self.device)

        self.optimizer.zero_grad()
        logits = self.model(inputs)

        loss = self.criterion(logits, target)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss}

    def val_step(self, inputs: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        inputs, target = inputs.to(self.device), target.to(self.device)

        self.optimizer.zero_grad()
        logits = self.model(inputs)
        loss = self.criterion(logits, target)
        _, predictions = torch.max(logits.data, 1)

        return {"loss": loss, "logits": logits, "predictions": predictions}

    def train_epoch(self, autodataset):
        train_dataloader = autodataset.train_dataloader
        tracker = self.tracker
        running_train_loss = 0.0
        tracker.train.steps = 0
        steps_per_epoch = tracker.steps_per_epoch

        tracker.train_prog = tracker.progress.add_task("[green]Learning...", total=len(train_dataloader))
        for step, data in enumerate(train_dataloader):
            inputs, target = data
            outputs = self.train_step(inputs, target)
            loss = outputs["loss"].item()
            running_train_loss += loss
            tracker.train.steps += 1
            tracker.progress.update(tracker.train_prog, advance=1)

            if self.TEST:
                break
            if steps_per_epoch and step >= steps_per_epoch:
                break
        tracker.train.loss = running_train_loss / (tracker.train.steps + 1e-9)
        tracker.progress.remove_task(tracker.train_prog)

    def val_epoch(self, autodataset):
        if not autodataset.val_dataloader:
            return
        val_dataloader = autodataset.val_dataloader
        tracker = self.tracker
        tracker.total = 0
        tracker.correct = 0
        running_val_loss = 0.0
        tracker.val.steps = 0

        val_prog = tracker.progress.add_task("[green]Validating...", total=len(val_dataloader))

        for _, data in enumerate(val_dataloader):
            with torch.no_grad():
                inputs, labels = data
                outputs = self.val_step(inputs, labels)
                loss = outputs["loss"]
                predicted = outputs["predictions"]
                tracker.total += labels.size(0)
                tracker.correct += (predicted == labels).sum().item()
                running_val_loss += loss.cpu().numpy()
                tracker.val.steps += 1
                tracker.progress.update(val_prog, advance=1)
            if self.TEST:
                break
        tracker.val.loss = running_val_loss / (tracker.val.steps + 1e-9)
        tracker.tune_metric = tracker.val_accuracy = tracker.correct / tracker.val.steps
        tracker.progress.remove_task(val_prog)

    def fit(
        self,
        autodataset: AutoDataset,
        epochs: int = 1,
        steps_per_epoch: Optional[int] = None,
        callbacks: Union[List, None] = None,
        resume: bool = True,
        progress_kwargs: Optional[Dict] = None,
    ) -> Tracker:
        """
        Similar to Keras model.fit() it trains the model for specified epochs and returns Tracker object
        Args:
            autodataset: AutoDataset object encapsulate dataloader and datamodule
            epochs: number of epochs to train
            steps_per_epoch: Number of steps trained in a single epoch
            callbacks: Callback object or string
            resume: Resume training from the last epoch
            progress_kwargs: Arguments for rich.progress

        Returns:
            Tracker object
        """
        optimizer = self.optimizer
        progress_kwargs = progress_kwargs or {}
        callbacks = listify(callbacks)

        if not resume:
            self.tracker.reset()
        tracker = self.tracker
        tracker.max_epochs = epochs
        tracker.optimizer = optimizer
        tracker.steps_per_epoch = steps_per_epoch
        callbacks = ComposeCallback(tracker, *callbacks)

        # ----- EVENT: ON_TRAINING_START
        callbacks.on_training_start()

        bar_column = BarColumn()
        table_column = RenderableColumn(tracker.create_table())

        progress = Progress(
            "[progress.description]{task.description}",
            bar_column,
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            table_column,
            expand=True,
            **progress_kwargs,
        )
        tracker.progress = progress
        with progress:
            tracker.epoch_prog = progress.add_task("[red]Epoch Progress...", total=epochs, completed=tracker.epoch)

            for epoch in range(tracker.epoch, epochs):
                tracker.epoch = epoch

                # ----- EVENT: ON_EPOCH_START
                callbacks.on_epoch_start()
                self.train_epoch(autodataset)
                table_column.renderable = tracker.create_table()

                # END OF TRAIN EPOCH
                self.val_epoch(autodataset)
                table_column.renderable = tracker.create_table()

                # ----- EVENT: ON_EPOCH_END
                callbacks.on_epoch_end()
                progress.update(tracker.epoch_prog, advance=1)

                if self.TEST:
                    break

        # ----- EVENT: ON_TRAINING_END
        callbacks.on_training_end()

        print("Finished Training")
        return tracker
