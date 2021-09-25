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

from typing import List, Union

import torch
from torch import nn

from gradsflow.core.callbacks import ComposeCallback, Tracker
from gradsflow.core.data import AutoDataset
from gradsflow.utility.common import listify, module_to_cls_index


class Model:
    _OPTIMIZER_INDEX = module_to_cls_index(torch.optim, True)

    def __init__(self, model: nn.Module, optimizer: str, lr: float = 3e-4):
        self.model = model
        self.lr = lr
        self.optimizer = self._OPTIMIZER_INDEX[optimizer](
            self.model.parameters(), lr=lr
        )

    def fit(
        self,
        autodataset: AutoDataset,
        epochs=1,
        callbacks: Union[List, None] = None,
        fast_dev_run: bool = False,
    ) -> Tracker:
        """
        Similar to Keras model.fit() it trains the model for specified epochs and returns Tracker object
        Args:
            autodataset: AutoDataset object encapsulate dataloader and datamodule
            epochs:
            callbacks:
            fast_dev_run: will run one step on train and val data (Inspired from Lightning)

        Returns:
            Tracker object
        """
        callbacks = listify(callbacks)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        train_dataloader = autodataset.train_dataloader
        val_dataloader = autodataset.val_dataloader

        model = self.model.to(device)
        optimizer = self.optimizer

        criterion = nn.CrossEntropyLoss()

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
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                tracker.running_loss += loss.item()
                tracker.train_loss += loss.item()
                tracker.epoch_steps += 1
                tracker.train_steps += 1
                if i % 100 == 0:  # print every 100 mini-batches
                    print(
                        f"epoch: {epoch}, loss: {tracker.running_loss / tracker.epoch_steps :.3f}"
                    )
                    tracker.running_loss = 0.0
                if fast_dev_run:
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
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    tracker.total += labels.size(0)
                    tracker.correct += (predicted == labels).sum().item()

                    loss = criterion(outputs, labels)
                    tracker.val_loss += loss.cpu().numpy()
                    tracker.val_steps += 1
                if fast_dev_run:
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
