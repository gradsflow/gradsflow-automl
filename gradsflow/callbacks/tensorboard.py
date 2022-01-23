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

from torch.utils.tensorboard import SummaryWriter

from gradsflow.callbacks.base import Callback

class TensorboardCallback(Callback):
    def __init__(self, 
                log_dir: str =None,
                comment: str ="",
                purge_step: int =None,
                max_queue: int =10,
                flush_secs: str =120,
                filename_suffix: str =""):
        super(TensorboardCallback, self).__init__()
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
            filename_suffix=self.filename_suffix)

    def on_train_epoch_end(self):
        tracker = self.model.tracker
        self.writer.add_scalar("train/loss", scalar_value=tracker.train.loss.avg, global_step=tracker.current_epoch)
        for metric, value in tracker.train.metrics.items():
            self.writer.add_scalar(f"train/{metric}", scalar_value=value.avg, global_step=tracker.current_epoch)

    def on_val_epoch_end(self):
        tracker = self.model.tracker
        self.writer.add_scalar("val/loss", scalar_value=tracker.val.loss.avg, global_step=tracker.current_epoch)
        for metric, value in tracker.val.metrics.items():
            self.writer.add_scalar(f"val/{metric}", scalar_value=value.avg, global_step=tracker.current_epoch)

    def on_fit_end(self):
        self.writer.close()
