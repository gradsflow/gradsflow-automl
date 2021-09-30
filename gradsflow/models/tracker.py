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

from rich.table import Table

from gradsflow.callbacks import ComposeCallback
from gradsflow.core.base import BaseTracker


class Tracker(BaseTracker):
    """
    Tracks loss, accuracy and model weights during model.fit()
    """

    def __init__(self):
        self.learner = None
        self.autodataset = None
        self.callbacks: ComposeCallback = None

    def create_table(self) -> Table:
        headings = ["epoch", "train/loss"]
        if self.val.loss:
            headings.append("val/loss")
        table = Table(*headings)

        row = [self.epoch, self.train.loss]
        if self.val.loss:
            row.append(self.val.loss)

        row = list(map(lambda x: f"{x: .3f}" if isinstance(x, float) else str(x), row))
        table.add_row(*row)
        return table

    def update_attributes(self, data: dict):
        """capture dictionary and update attributes of tracker"""

        for k, v in data.items():
            if getattr(self, k):
                setattr(self, k, v)
