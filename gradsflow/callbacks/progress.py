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
from typing import Dict, Optional

from rich.progress import BarColumn, Progress, RenderableColumn, TimeRemainingColumn

from gradsflow.core.callbacks import Callback


class ProgressCallback(Callback):
    _name = "ProgressCallback"

    def __init__(self, model, progress_kwargs: Optional[Dict] = None):
        super().__init__(model)
        progress_kwargs = progress_kwargs or {}
        tracker = self.model.tracker
        self.bar_column = BarColumn()
        self.table_column = RenderableColumn(tracker.create_table())

        self.progress = Progress(
            "[progress.description]{task.description}",
            self.bar_column,
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            self.table_column,
            **progress_kwargs,
        )
        tracker.progress = self.progress
        self.fit_prog = None
        self.train_prog_bar = None
        self.val_prog_bar = None

    def on_fit_start(self):
        self.progress.start()
        epochs = self.model.tracker.max_epochs
        completed = self.model.tracker.current_epoch
        self.fit_prog = self.progress.add_task("[red]Progress", total=epochs, completed=completed)

    def on_fit_end(self):
        self.progress.stop()

    def on_epoch_end(self):
        self.progress.update(self.fit_prog, advance=1)

    def on_train_epoch_start(self):
        n = len(self.model.autodataset.train_dataloader)
        self.train_prog_bar = self.progress.add_task("[green]Learning", total=n)

    def on_train_epoch_end(self, *args, **kwargs):
        self.progress.remove_task(self.train_prog_bar)
        self.table_column.renderable = self.model.tracker.create_table()

    def on_train_step_end(self, *args, **kwargs):
        self.progress.update(self.train_prog_bar, advance=1)
        self.table_column.renderable = self.model.tracker.create_table()

    def on_val_epoch_start(self):
        val_dl = self.model.autodataset.val_dataloader
        if not val_dl:
            return
        n = len(val_dl)
        self.val_prog_bar = self.progress.add_task("[blue]Validating...", total=n)

    def on_val_epoch_end(self, *args, **kwargs):
        val_dl = self.model.autodataset.val_dataloader
        if not val_dl:
            return
        self.table_column.renderable = self.model.tracker.create_table()
        self.progress.remove_task(self.val_prog_bar)

    def on_val_step_end(self, *args, **kwargs):
        self.progress.update(self.val_prog_bar, advance=1)
        self.table_column.renderable = self.model.tracker.create_table()

    def clean(self):
        self.progress.stop()
