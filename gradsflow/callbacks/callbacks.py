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
    def __init__(self, model: "Model"):
        self.model = model

    def on_fit_start(self):
        ...

    def on_fit_end(
        self,
    ):
        ...

    def on_train_epoch_start(
        self,
    ):
        ...

    def on_train_epoch_end(self):
        ...

    def on_val_epoch_start(
        self,
    ):
        ...

    def on_val_epoch_end(self):
        ...

    def on_train_step_start(self):
        ...

    def on_train_step_end(self):
        ...

    def on_val_step_start(self):
        ...

    def on_val_step_end(self):
        ...

    def on_epoch_start(self):
        ...

    def on_epoch_end(self):
        ...
