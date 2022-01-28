#  Copyright (c) 2022 GradsFlow. All rights reserved.
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

from typing import Union

from torch.utils.tensorboard import SummaryWriter

from gradsflow.callbacks.base import Callback
from gradsflow.utility.imports import requires


class TensorboardCallback(Callback):
    """
    Tensorboard callback. To use this callback `pip install tensorboard`.
    Metrics and losses for all `train_step`'s and for all `global_step`'s are plotted on tensorboard.
    Inspired from PyTorch.
    Ref: https://pytorch.org/docs/stable/tensorboard.html

    Args:
        log_dir: Save directory location. Default is runs/CURRENT_DATETIME_HOSTNAME, which changes after each run.
                Use hierarchical folder structure to compare between runs easily. e.g. pass in "runs/exp1", "runs/exp2",
                etc. for each new experiment to compare across them
        comment: Comment log_dir suffix appended to the default log_dir. If log_dir is assigned, this argument has no effect
        purge_step: When logging crashes at step T+XT+X and restarts at step TT, any events whose global_step larger or equal
                    to TT will be purged and hidden from TensorBoard. Note that crashed and resumed experiments should have
                    the same log_dir
        max_queue: Size of the queue for pending events and summaries before one of the "add" calls forces a flush to disk.
                    Default is ten items
        flush_secs: How often, in seconds, to flush the pending events and summaries to disk. Default is every two minutes
                    filename_suffix: Suffix added to all event filenames in the log_dir directory. More details on filename construction
                    in tensorboard.summary.writer.event_file_writer.EventFileWriter

    ```python
    from gradsflow.callbacks import TensorboardCallback
    from timm import create_model

    cnn = create_model("resnet18", pretrained=False, num_classes=1)
    model = Model(cnn)
    model.compile()
    tb = TensorboardCallback("logs/resnet")
    autodataset = None # create your dataset
    model.fit(autodataset, callbacks=tb)
    ```
    """

    @requires("tensorboard", "TensorboardCallback requires tensorboard to be installed!")
    def __init__(
        self,
        log_dir: str = None,
        comment: str = "",
        purge_step: int = None,
        max_queue: int = 10,
        flush_secs: str = 120,
        filename_suffix: str = "",
    ):
        super().__init__()
        self.log_dir = log_dir
        self.comment = comment
        self.purge_step = purge_step
        self.max_queue = max_queue
        self.flush_secs = flush_secs
        self.filename_suffix = filename_suffix
        self.writer = None
        self.writer = SummaryWriter(
            log_dir=self.log_dir,
            comment=self.comment,
            purge_step=self.purge_step,
            flush_secs=self.flush_secs,
            filename_suffix=self.filename_suffix,
        )

    def _add_scalar(self, tag: str, scalar_value: Union[float, int], global_step: int):
        self.writer.add_scalar(tag, scalar_value=scalar_value, global_step=global_step)

    def on_train_epoch_end(self):
        prefix = "train"
        loss = self.model.tracker.train_loss
        epoch = self.model.tracker.current_epoch
        # Plotting train epoch_loss
        self._add_scalar(f"{prefix}/epoch_loss", loss, epoch)

        # Plotting train epoch_metrics
        metrics = self.model.tracker.train_metrics.to_dict()
        for metric, value in metrics.items():
            self._add_scalar(f"{prefix}/epoch_{metric}", value["avg"], epoch)

    def on_val_epoch_end(self):
        prefix = "val"
        loss = self.model.tracker.val_loss
        epoch = self.model.tracker.current_epoch
        # Plotting val epoch_loss
        self._add_scalar(f"{prefix}/epoch_loss", loss, epoch)

        # Plotting val epoch_metrics
        metrics = self.model.tracker.val_metrics.to_dict()
        for metric, value in metrics.items():
            self._add_scalar(f"{prefix}/epoch_{metric}", value["avg"], epoch)

    def on_train_step_end(self, outputs: dict = None, **_):
        prefix = "train"
        loss = outputs["loss"].item()
        global_step = self.model.tracker.global_step
        # Plotting train step_loss
        self._add_scalar(f"{prefix}/step_loss", loss, global_step)

        # Plotting train step_metrics
        metrics = outputs.get("metrics", {})
        for metric, value in metrics.items():
            self._add_scalar(f"{prefix}/step_{metric}", value, global_step)

    def on_val_step_end(self, outputs: dict = None, **_):
        prefix = "val"
        loss = outputs["loss"].item()
        global_step = self.model.tracker.global_step
        # Plotting val step_loss
        self._add_scalar(f"{prefix}/step_loss", loss, global_step)

        # Plotting val step_metrics
        metrics = outputs.get("metrics", {})
        for metric, value in metrics.items():
            self._add_scalar(f"{prefix}/step_{metric}", value, global_step)

    def on_fit_end(self):
        self.writer.close()
