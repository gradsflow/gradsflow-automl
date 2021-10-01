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
from typing import Optional

from rich.table import Table

from gradsflow.callbacks import CallbackRunner
from gradsflow.core.base import BaseTracker


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

    def create_table(self) -> Table:
        headings = ["epoch", "train/loss"]
        row = [self.epoch, self.train.loss]
        if self.val.loss:
            headings.append("val/loss")

        for metric_name, _ in self.train.metrics.items():
            headings.append("train/" + metric_name)

        for metric_name, _ in self.val.metrics.items():
            headings.append("val/" + metric_name)

        if self.val.loss:
            row.append(self.val.loss)

        for _, value in self.train.metrics.items():
            row.append(value)

        for _, value in self.val.metrics.items():
            row.append(value)

        row = list(map(lambda x: f"{x: .3f}" if isinstance(x, float) else str(x), row))
        table = Table(*headings, expand=False)
        table.add_row(*row)
        return table
