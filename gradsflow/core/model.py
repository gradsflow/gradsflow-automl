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
from typing import Dict, List, Union

import torch
from torch import nn

from gradsflow.core.callbacks import ComposeCallback, Tracker
from gradsflow.core.data import AutoDataset
from gradsflow.utility.common import listify, module_to_cls_index


class Model:
    TEST = os.environ.get("GF_CI")
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

    def fit(
        self,
        autodataset: AutoDataset,
        epochs=1,
        callbacks: Union[List, None] = None,
    ) -> Tracker:
        """
        Similar to Keras model.fit() it trains the model for specified epochs and returns Tracker object
        Args:
            autodataset: AutoDataset object encapsulate dataloader and datamodule
            epochs:
            callbacks:

        Returns:
            Tracker object
        """
        device = self.device
        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion

        callbacks = listify(callbacks)
        train_dataloader = autodataset.train_dataloader
        val_dataloader = autodataset.val_dataloader

        tracker = Tracker()
        tracker.model = model
        tracker.optimizer = optimizer
        callbacks = ComposeCallback(tracker, *callbacks)

        # ----- EVENT: ON_TRAINING_START
        callbacks.on_training_start()

        for epoch in range(epochs):  # loop over the dataset multiple times
            tracker.epoch = epoch
            tracker.running_loss = 0.0
            tracker.epoch_steps = 0
            tracker.train_loss = 0.0
            tracker.train_steps = 0

            # ----- EVENT: ON_EPOCH_START
            callbacks.on_epoch_start()

            for i, data in enumerate(train_dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, target = data
                outputs = self.train_step(inputs, target)
                loss = outputs["loss"]

                # print statistics
                loss = loss.item()
                tracker.running_loss += loss
                tracker.train_loss += loss
                tracker.epoch_steps += 1
                tracker.train_steps += 1
                if i % 100 == 0:  # print every 100 mini-batches
                    print(
                        f"epoch: {epoch}, loss: {tracker.running_loss / tracker.epoch_steps :.3f}"
                    )
                    tracker.running_loss = 0.0
                if self.TEST:
                    break
            # END OF TRAIN EPOCH
            tracker.train_loss /= tracker.train_steps + 1e-9

            # Validation loss
            tracker.val_loss = 0.0
            tracker.val_steps = 0
            tracker.total = 0
            tracker.correct = 0
            for i, data in enumerate(val_dataloader, 0):
                with torch.no_grad():
                    inputs, labels = data
                    outputs = self.val_step(inputs, labels)
                    loss = outputs["loss"]
                    predicted = outputs["predictions"]

                    tracker.total += labels.size(0)
                    tracker.correct += (predicted == labels).sum().item()

                    tracker.val_loss += loss.cpu().numpy()
                    tracker.val_steps += 1
                if self.TEST:
                    break
            tracker.val_loss /= tracker.val_steps + 1e-9
            tracker.val_accuracy = tracker.correct / tracker.val_steps

            print(
                f"epoch {tracker.epoch}: train/loss={tracker.train_loss}, "
                f"val/loss={tracker.val_loss}, val/accuracy={tracker.val_accuracy}"
            )

            # ----- EVENT: ON_EPOCH_END
            callbacks.on_epoch_end()

        # ----- EVENT: ON_TRAINING_END
        callbacks.on_epoch_end()

        print("Finished Training")
        return tracker

    def load_from_checkpoint(self, checkpoint):
        self.model = torch.load(checkpoint)
