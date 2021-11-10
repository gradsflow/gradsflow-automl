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

from gradsflow.callbacks.callbacks import Callback


class ModelCheckpoint(Callback):
    def __init__(self, filename: Optional[str] = None, path: str = os.getcwd(), save_extra: bool = False):
        """
        Saves Model checkpoint
        Args:
            filename: name of checkpoint
            path: folder path location of the model checkpoint
            save_extra: whether to save extra details like tracker
        """
        super().__init__(model=None)
        filename = filename or "model"
        self.path = path
        self._dst = Path(path) / Path(filename)
        self.save_extra = save_extra

    def on_epoch_end(self):
        epoch = self.model.tracker.current_epoch
        path = f"{self._dst}_epoch={epoch}_.pt"
        self.model.save(path, save_extra=self.save_extra)
