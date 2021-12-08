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
from abc import ABC
from typing import Callable, Optional

if typing.TYPE_CHECKING:
    from gradsflow.models.model import Model


def dummy(x=None, **__):
    return x


class Callback(ABC):
    """Callback objects define events on which it will run during the model training cycle."""

    _events = ("forward", "step", "train_epoch", "val_epoch", "epoch", "fit")
    _name: str = "Callback"

    def __init__(self, model: Optional["Model"] = None):
        self.model = model

    def with_event(self, event_type: str, func: Callable, exception, final_fn: Callable = dummy):
        """Calls a function with event wrapped around. Inspired from FastAI.
        Ref: https://github.com/fastai/fastai/blob/6e44b354f4d12bdfa2c9530f38f851c54a05764d/fastai/learner.py#L162
        """
        assert event_type in self._events, f"event_type is {event_type} but should be {self._events}"
        start_event = f"on_{event_type}_start"
        end_event = f"on_{event_type}_end"
        cancel_event = f"on_{event_type}_cancel"
        try:
            getattr(self, start_event)()
            func()
        except exception:
            getattr(self, cancel_event)()
        getattr(self, end_event)()
        final_fn()

    def on_fit_start(self):
        """Called on each `model.fit(...)`"""

    def on_fit_end(
        self,
    ):
        """Called after `model.fit(...)`"""

    def on_fit_cancel(
        self,
    ):
        """Called after `model.fit(...)`is cancelled"""

    def on_train_epoch_start(
        self,
    ):
        """Called on start of training epoch"""

    def on_train_epoch_end(self, *args, **kwargs):
        """Called after end of training epoch"""

    def on_train_epoch_cancel(self):
        """Called after training epoch is cancelled"""

    def on_val_epoch_start(
        self,
    ):
        """Called on start of validation epoch"""

    def on_val_epoch_end(self, *args, **kwargs):
        """called after validation epoch ends"""

    def on_val_epoch_cancel(self):
        """called after validation epoch cancelled"""

    def on_train_step_start(self):
        """called before `train_step`"""

    def on_train_step_end(self, *args, **kwargs):
        """Called after training step"""

    def on_train_step_cancel(self):
        """Called after training step is cancelled"""

    def on_val_step_start(self):
        """Called on validation step"""

    def on_val_step_end(self, *args, **kwargs):
        """Called after validation step"""

    def on_val_step_cancel(self):
        """Called after validation step is cancelled"""

    def on_epoch_start(self):
        """Called Before each Epoch"""

    def on_epoch_end(self):
        """Called after each epoch"""

    def on_epoch_cancel(self):
        """Called after epoch is cancelled"""

    def on_forward_start(self):
        """Called before model.forward(...)"""

    def on_forward_end(self):
        """Called after model.forward(...)"""

    def on_forward_cancel(self):
        """Called after model.forward(...) is cancelled"""

    def clean(self):
        """Clean up"""
