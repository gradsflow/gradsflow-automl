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
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from gradsflow.core.callbacks import Callback
from gradsflow.utility.common import to_item


class CSVLogger(Callback):
    """
    Saves Model training metrics as CSV
    Args:
        filename: filename of the csv
        path: folder path location of the csv
        verbose: Whether to show output
    """

    _name = "CSVLogger"

    def __init__(self, filename: str = "./experiment.csv", path: str = os.getcwd(), verbose: bool = False):
        super().__init__(model=None)
        self.filename = filename
        self.path = path
        self._dst = Path(path) / Path(filename)
        self._logs = []
        self.verbose = verbose

    def on_epoch_end(self):
        epoch = self.model.tracker.current_epoch
        train_loss = self.model.tracker.train_loss
        val_loss = self.model.tracker.val_loss
        train_metrics = self.model.tracker.train_metrics
        val_metrics = self.model.tracker.val_metrics

        train_metrics = {"train/" + k: v.avg for k, v in train_metrics.items()}
        val_metrics = {"val/" + k: v.avg for k, v in val_metrics.items()}
        train_metrics = to_item(train_metrics)
        val_metrics = to_item(val_metrics)

        data = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **train_metrics, **val_metrics}
        if self.verbose:
            logger.info(f"verbose csv_logger on_epoch_end: {data}")
        self._logs.append(data)
        df = pd.DataFrame(self._logs)
        df.to_csv(self._dst, index=False)


class ModelCheckpoint(Callback):
    """
    Saves Model checkpoint
    Args:
        filename: name of checkpoint
        path: folder path location of the model checkpoint
        save_extra: whether to save extra details like tracker
    """

    _name = "ModelCheckpoint"

    def __init__(self, filename: Optional[str] = None, path: str = os.getcwd(), save_extra: bool = False):
        super().__init__(model=None)
        filename = filename or "model"
        self.path = path
        self._dst = Path(path) / Path(filename)
        self.save_extra = save_extra

    def on_epoch_end(self):
        epoch = self.model.tracker.current_epoch
        path = f"{self._dst}_epoch={epoch}_.pt"
        self.model.save(path, save_extra=self.save_extra)
