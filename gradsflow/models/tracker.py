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
from dataclasses import dataclass
from typing import Dict, List, Optional

from loguru import logger
from rich import box
from rich.table import Table

from gradsflow.core.base import TrackingValues
from gradsflow.utility.common import GDict, to_item


@dataclass(init=False)
class BaseTracker:
    global_step: int = 0  # Global training steps
    max_epochs: int = 0
    current_epoch: int = 0  # current train current_epoch
    steps_per_epoch: Optional[int] = None
    train: TrackingValues = TrackingValues()
    val: TrackingValues = TrackingValues()


class Tracker(BaseTracker):
    """
    Tracks loss, accuracy and model weights during model.fit()
    """

    def __init__(self):
        self.train.metrics = GDict()
        self.val.metrics = GDict()
        self.logs: List[Dict] = []

    def __getitem__(self, key: str):  # skipcq: PYL-R1705
        """
        1. key= `train | val` then return respective `TrackingValues` object
        2. key=`metrics` then return a dictionary of metrics
        3. key=`loss` then return a dictionary of losses
        Args:
            key: train, val, metrics or loss

        Returns:
            `TrackingValues` or a Dictionary
        """
        if key in ("train", "val"):
            return self.mode(key)
        elif key == "metrics":
            return {"train": self.train_metrics, "val": self.val_metrics}
        elif key == "loss":
            return {"train": self.train_loss, "val": self.val_loss}

        raise KeyError(f"key {key} is not implemented!")

    @property
    def train_loss(self):
        return self.train.loss.avg

    @property
    def val_loss(self):
        return self.val.loss.avg

    @property
    def train_metrics(self) -> GDict:
        return self.train.metrics

    @property
    def val_metrics(self) -> GDict:
        return self.val.metrics

    def mode(self, mode) -> TrackingValues:
        if mode == "train":
            return self.train
        if mode == "val":
            return self.val

        raise KeyError(f"mode {mode} is not implemented!")

    def _append_logs(self, key, value):
        """Append Key Value pairs to `Tracker.logs`"""
        # TODO: accept a list of keys and values as well.
        epoch = self.current_epoch
        data = {"current_epoch": epoch, key: to_item(value)}
        self.logs.append(data)

    def track_loss(self, loss: float, mode: str):
        """Tracks loss by adding to `Tracker.logs` and maintaining average loss in a single Epoch with `TrackingValues`.
        Update loss with `TrackingValues.update_loss(loss)` which is called with `TrainEvalCallback` at `*_step_end`.
        Args:
            loss: Step Loss
            mode: can be train | val
        """
        loss = to_item(loss)
        value_tracker = self.mode(mode)
        value_tracker.update_loss(loss)
        key = mode + "/" + "loss"
        self._append_logs(key, loss)

    def track_metrics(self, metric: Dict[str, float], mode: str):
        """Tracks metrics by adding to `Tracker.logs` and maintaining average metric in a single Epoch with `TrackingValues`.
        Update  metrics with `TrackingValues.update_metrics(metrics)` which is called with `TrainEvalCallback` at `*_step_end`.
        Args:
            metric: Step metric
            mode: can be train | val
        """
        value_tracker = self.mode(mode)

        # Track values that averages with epoch
        value_tracker.update_metrics(metric)

        # _append_logs value for each step in a dict
        for k, v in metric.items():
            k = mode + "/" + k
            self._append_logs(k, v)

    def create_table(self) -> Table:
        headings = ["i", "train/loss"]
        row = [self.current_epoch, self.train_loss]

        if self.val.loss.computed:
            headings.append("val/loss")
            row.append(self.val_loss)

        for metric_name, value in self.train_metrics.items():
            headings.append("train/" + metric_name)
            row.append(value.avg)

        for metric_name, value in self.val_metrics.items():
            headings.append("val/" + metric_name)
            row.append(value.avg)

        row = list(map(lambda x: f"{x: .3f}" if isinstance(x, float) else str(x), row))
        table = Table(*headings, expand=True, box=box.SIMPLE)
        table.add_row(*row)
        return table

    def reset(self):
        """Resets epochs, logs and train & val `TrackingValues`."""
        logger.debug("Reset Tracker")
        self.max_epochs = 0
        self.current_epoch = 0
        self.steps_per_epoch = None
        self.train = TrackingValues()
        self.val = TrackingValues()
        self.logs = []
