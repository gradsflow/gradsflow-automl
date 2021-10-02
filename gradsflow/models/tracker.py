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
import dataclasses
from typing import Dict, List, Optional

import pandas as pd
from rich.table import Table

from gradsflow.callbacks import CallbackRunner
from gradsflow.core.base import BaseTracker, TrackingValues
from gradsflow.models.utils import to_item


class Tracker(BaseTracker):
    """
    Tracks loss, accuracy and model weights during model.fit()
    """

    def __init__(self):
        self.learner = None
        self.autodataset = None
        self.callback_runner: Optional[CallbackRunner] = None
        self.train.metrics = {}
        self.val.metrics = {}
        self.logs: List[Dict] = []
        self.non_render_logs: List[Dict] = []

    def mode(self, mode) -> TrackingValues:
        if mode == "train":
            return self.train
        if mode == "val":
            return self.val

        raise NotImplementedError(f"mode {mode} is not implemented!")

    def track(self, key, value, render=False):
        epoch = self.current_epoch
        step = self.current_step
        data = {"current_epoch": epoch, "current_step": step, key: to_item(value)}
        if render:
            self.logs.append(data)
        else:
            self.non_render_logs.append(data)

    def track_loss(self, loss: float, mode: str):
        """Update `TrackingValues` loss. mode can be train or val"""
        value_tracker = self.mode(mode)
        value_tracker.loss = loss
        key = mode + "/" + "loss"
        self.track(key, loss, render=True)

    def track_metrics(self, metric: Dict[str, float], mode: str, render: bool = False):
        """Update `TrackingValues` metrics. mode can be train or val and will update logs if render is True"""
        value_tracker = self.mode(mode)
        value_tracker.metrics = metric
        if not render:
            return
        for k, v in metric.items():
            k = mode + "/" + k
            self.track(k, v, render=render)

    def get_metrics(self, mode: str):
        value_tracker = self.mode(mode)
        return value_tracker.metrics

    def get_loss(self, mode: str):
        value_tracker = self.mode(mode)
        return value_tracker.loss

    def create_table(self) -> Table:
        headings = ["current_epoch", "train/loss"]
        row = [self.current_epoch, to_item(self.train.loss)]
        if self.val.loss:
            headings.append("val/loss")

        for metric_name, _ in self.train.metrics.items():
            headings.append("train/" + metric_name)

        for metric_name, _ in self.val.metrics.items():
            headings.append("val/" + metric_name)

        if self.val.loss:
            row.append(to_item(self.val.loss))

        for _, value in self.train.metrics.items():
            row.append(to_item(value))

        for _, value in self.val.metrics.items():
            row.append(to_item(value))

        row = list(map(lambda x: f"{x: .3f}" if isinstance(x, float) else str(x), row))
        table = Table(*headings, expand=False)
        table.add_row(*row)
        return table

    def reset(self):
        self.max_epochs = 0
        self.current_epoch = 0
        self.current_step = 0
        self.steps_per_epoch = None
        self.train = TrackingValues()
        self.val = TrackingValues()
        self.logs = []
        self.non_render_logs = []
