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
from rich.progress import Progress
from torch import nn

from gradsflow.core.callbacks import ComposeCallback, Tracker
from gradsflow.core.data import AutoDataset
from gradsflow.utility.common import listify, module_to_cls_index


class Model:
    TEST = os.environ.get("GF_CI", "false").lower() == "true"
    _OPTIMIZER_INDEX = module_to_cls_index(torch.optim, True)

    def __init__(self, model: nn.Module, optimizer: str, lr: float = 3e-4):
        self.model = model
        self.lr = lr
        self.optimizer = self._OPTIMIZER_INDEX[optimizer](
            self.model.parameters(), lr=lr
        )

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.tracker = Tracker()
        self.tracker.model = self.model

    def __call__(self, inputs):
        return self.model(inputs)

    @torch.no_grad()
    def predict(self, inputs):
        return self.model(inputs)

    def train_step(
        self, inputs: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        inputs, target = inputs.to(self.device), target.to(self.device)

        self.optimizer.zero_grad()
        logits = self.model(inputs)

        loss = self.criterion(logits, target)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss}

    def val_step(
        self, inputs: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
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
        tracker.running_loss = 0.0
        tracker.train_steps = 0

        for step, data in enumerate(train_dataloader):
            inputs, target = data
            outputs = self.train_step(inputs, target)
            loss = outputs["loss"]

            loss = loss.item()
            tracker.running_loss += loss
            running_train_loss += loss
            tracker.train_steps += 1
            steps_per_epoch = tracker.steps_per_epoch

            if step % 100 == 0:  # print every 100 mini-batches
                print(
                    f"epoch: {tracker.epoch}, "
                    f"loss: {tracker.running_loss / tracker.train_steps :.3f}"
                )
                tracker.running_loss = 0.0
            if self.TEST:
                break
            if steps_per_epoch and step >= steps_per_epoch:
                break
        tracker.train_loss = running_train_loss / (tracker.train_steps + 1e-9)
        print(f"epoch: {tracker.epoch}: train/loss={tracker.train_loss: .3f}")

    def val_epoch(self, autodataset):
        if not autodataset.val_dataloader:
            return
        val_dataloader = autodataset.val_dataloader
        tracker = self.tracker
        tracker.total = 0
        tracker.correct = 0
        running_val_loss = 0.0
        tracker.val_steps = 0

        val_prog = tracker.progress.add_task(
            "[green]Validation...", total=len(val_dataloader)
        )

        for _, data in enumerate(val_dataloader):
            with torch.no_grad():
                inputs, labels = data
                outputs = self.val_step(inputs, labels)
                loss = outputs["loss"]
                predicted = outputs["predictions"]

                tracker.total += labels.size(0)
                tracker.correct += (predicted == labels).sum().item()

                running_val_loss += loss.cpu().numpy()
                tracker.val_steps += 1
                tracker.progress.update(val_prog, advance=1)
            if self.TEST:
                break
        tracker.val_loss = running_val_loss / (tracker.val_steps + 1e-9)
        tracker.val_accuracy = tracker.correct / tracker.val_steps
        print(
            f"val/loss={tracker.val_loss: .3f},"
            f" val/accuracy={tracker.val_accuracy: .3f}"
        )
        tracker.progress.remove_task(val_prog)

    def fit(
        self,
        autodataset: AutoDataset,
        epochs: int = 1,
        steps_per_epoch: Optional[int] = None,
        callbacks: Union[List, None] = None,
        progress_kwargs: Optional[Dict] = None,
    ) -> Tracker:
        """
        Similar to Keras model.fit() it trains the model for specified epochs and returns Tracker object
        Args:
            autodataset: AutoDataset object encapsulate dataloader and datamodule
            epochs: number of epochs to train
            steps_per_epoch: Number of steps trained in a single epoch
            callbacks: Callback object or string
            progress_kwargs: Arguments for rich.progress

        Returns:
            Tracker object
        """
        optimizer = self.optimizer
        progress_kwargs = progress_kwargs or {}

        callbacks = listify(callbacks)

        tracker = self.tracker
        tracker.max_epochs = epochs
        tracker.optimizer = optimizer
        tracker.steps_per_epoch = steps_per_epoch
        callbacks = ComposeCallback(tracker, *callbacks)

        # ----- EVENT: ON_TRAINING_START
        callbacks.on_training_start()

        with Progress(**progress_kwargs) as progress:
            tracker.progress = progress
            train_prog = progress.add_task("[blue]Training...", total=epochs)

            for epoch in range(tracker.epoch, epochs):
                tracker.epoch = epoch

                # ----- EVENT: ON_EPOCH_START
                callbacks.on_epoch_start()
                self.train_epoch(autodataset)
                progress.update(train_prog, advance=1)

                # END OF TRAIN EPOCH
                self.val_epoch(autodataset)

                # ----- EVENT: ON_EPOCH_END
                callbacks.on_epoch_end()

                if self.TEST:
                    break

        # ----- EVENT: ON_TRAINING_END
        callbacks.on_epoch_end()

        print("Finished Training")
        return tracker

    def load_from_checkpoint(self, checkpoint):
        self.model = torch.load(checkpoint)
