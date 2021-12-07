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
from typing import Dict, List

from loguru import logger
from rich import box
from rich.table import Table

from gradsflow.core.base import BaseTracker, TrackingValues
from gradsflow.utility.common import AverageMeter, to_item


class Tracker(BaseTracker):
    """
    Tracks loss, accuracy and model weights during model.fit()
    """

    def __init__(self):
        self.train.metrics = {}
        self.val.metrics = {}
        self.logs: List[Dict] = []

    def mode(self, mode) -> TrackingValues:
        if mode == "train":
            return self.train
        if mode == "val":
            return self.val

        raise NotImplementedError(f"mode {mode} is not implemented!")

    def track(self, key, value):
        """Tracks value"""
        epoch = self.current_epoch
        data = {"current_epoch": epoch, key: to_item(value)}
        self.logs.append(data)

    def track_loss(self, loss: float, mode: str):
        """Update `TrackingValues` loss. mode can be train or val"""
        value_tracker = self.mode(mode)
        value_tracker.loss.update(loss)
        key = mode + "/" + "loss"
        self.track(key, loss)

    def track_metrics(self, metric: Dict[str, float], mode: str):
        """Update `TrackingValues` metrics. mode can be train or val"""
        value_tracker = self.mode(mode)
        # Track values that averages with epoch
        for key, value in metric.items():
            try:
                value_tracker.metrics[key].update(value)
            except KeyError:
                value_tracker.metrics[key] = AverageMeter(name=key)
                value_tracker.metrics[key].update(value)

        # track value for each step in a dict
        for k, v in metric.items():
            k = mode + "/" + k
            self.track(k, v)

    def get_metrics(self, mode: str):
        value_tracker = self.mode(mode)
        return value_tracker.metrics

    def get_loss(self, mode: str):
        value_tracker = self.mode(mode)
        return value_tracker.loss.avg

    def create_table(self) -> Table:
        headings = ["i", "train/loss"]
        row = [self.current_epoch, to_item(self.train_loss)]

        if self.val.loss.computed:
            headings.append("val/loss")
            row.append(to_item(self.val_loss))

        for metric_name, value in self.train_metrics.items():
            headings.append("train/" + metric_name)
            row.append(to_item(value.avg))

        for metric_name, value in self.val_metrics.items():
            headings.append("val/" + metric_name)
            row.append(to_item(value.avg))

        row = list(map(lambda x: f"{x: .3f}" if isinstance(x, float) else str(x), row))
        table = Table(*headings, expand=True, box=box.SIMPLE)
        table.add_row(*row)
        return table

    def reset(self):
        logger.info("Reset Tracker")
        self.max_epochs = 0
        self.current_epoch = 0
        self.steps_per_epoch = None
        self.train = TrackingValues()
        self.val = TrackingValues()
        self.logs = []

    @property
    def train_loss(self):
        return self.train.loss.avg

    @property
    def val_loss(self):
        return self.val.loss.avg

    @property
    def train_metrics(self):
        return self.train.metrics

    @property
    def val_metrics(self):
        return self.val.metrics
