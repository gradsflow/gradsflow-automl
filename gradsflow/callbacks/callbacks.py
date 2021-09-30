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
import typing

if typing.TYPE_CHECKING:
    from gradsflow.models.model import Model


class Callback:
    """Callback objects define events on which it will run during the model training cycle."""

    def __init__(self, model: "Model"):
        self.model = model

    def on_fit_start(self):
        """Called on each `model.fit(...)`"""

    def on_fit_end(
        self,
    ):
        """Called after `model.fit(...)`"""

    def on_train_epoch_start(
        self,
    ):
        """Called on start of training epoch"""

    def on_train_epoch_end(self):
        """Called after end of training epoch"""

    def on_val_epoch_start(
        self,
    ):
        """Called on start of validation epoch"""

    def on_val_epoch_end(self):
        """called after validation epoch ends"""

    def on_train_step_start(self):
        """called before `train_step`"""

    def on_train_step_end(self):
        """Called after training step"""

    def on_val_step_start(self):
        """Called on validation step"""

    def on_val_step_end(self):
        """Called after validation step"""

    def on_epoch_start(self):
        """Called Before each training Epoch"""

    def on_epoch_end(self):
        """Called after each training epoch"""
