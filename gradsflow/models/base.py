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
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Union

import smart_open
import torch
from lightning.fabric import Fabric
from torch import nn

from gradsflow.models.tracker import Tracker
from gradsflow.models.utils import losses
from gradsflow.utility.common import default_device, module_to_cls_index

_OPTIMIZER_INDEX = module_to_cls_index(torch.optim, True)


@dataclass(init=False)
class Base:
    TEST = os.environ.get("GF_CI", "false").lower() == "true"

    _learner: Union[nn.Module, Any]
    optimizer: torch.optim.Optimizer = None
    loss: Callable = None
    _compiled: bool = False

    def __init__(self):
        self.tracker = Tracker()
        self.device = None

    def __call__(self, x):
        return self.forward(x)

    @property
    def learner(self) -> Union[nn.Module, Any]:
        return self._learner

    @learner.setter
    def learner(self, learner):
        self._learner = learner

    @staticmethod
    def _get_loss(loss: Union[str, Callable], loss_config: dict) -> Optional[Callable]:
        loss_fn = None
        if isinstance(loss, str):
            loss_fn = losses.get(loss)(**loss_config)
            assert loss_fn is not None, f"loss {loss} is not available! Available losses are {tuple(losses.keys())}"
        elif isinstance(loss, type):  # when loss is a class
            loss_fn = loss(**loss_config)
        elif callable(loss):
            loss_fn = loss

        return loss_fn

    @staticmethod
    def _get_optimizer(optimizer: Union[str, torch.optim.Optimizer]) -> Callable:
        if isinstance(optimizer, str):
            optimizer_fn = _OPTIMIZER_INDEX.get(optimizer)
            assert (
                optimizer_fn
            ), f"optimizer {optimizer} is not available! Available optimizers are {tuple(_OPTIMIZER_INDEX.keys())}"

        elif callable(optimizer):
            assert optimizer in tuple(_OPTIMIZER_INDEX.values()), f"Unknown Optimizer {type(optimizer)}"
            optimizer_fn = optimizer
        else:
            raise NotImplementedError(f"Unknown optimizer {optimizer}")
        return optimizer_fn

    def assert_compiled(self):
        if not self._compiled:
            raise UserWarning("Model not compiled yet! Please call `model.compile(...)` first.")

    @torch.no_grad()
    def predict(self, x):
        return self.learner(x)

    def forward(self, x):
        return self.learner(x)

    # skipcp: PTC-W0049
    def backward(self, loss):
        ...

    # skipcp: PTC-W0049
    def eval(self):
        ...

    # skipcp: PTC-W0049
    def train(self):
        ...


class BaseModel(Base):
    """Base Class of Model API implemented with HF Accelerate"""

    def __init__(
        self,
        learner: Union[nn.Module, Any],
        device: Optional[str] = "auto",
        strategy: Optional[str] = None,
        precision: Any = 32,
        num_nodes: int = 1,
        use_accelerator: bool = True,
        accelerator_config: dict = None,
    ):
        self._accelerator = None
        super().__init__()
        self._set_accelerator(device, strategy, precision, num_nodes, use_accelerator, accelerator_config)
        self._learner = learner

    def _set_accelerator(
        self, device: Optional[str], strategy, precision, num_nodes, use_accelerate: bool, accelerator_config: dict
    ):
        if use_accelerate:
            self._accelerator = Fabric(
                accelerator=device, strategy=strategy, precision=precision, num_nodes=num_nodes, **accelerator_config
            )
            self.device = self._accelerator.device
        else:
            self.device = device or default_device()

    def setup(self, learner: Union[nn.Module, List[nn.Module]], *optimizers):
        if not self._accelerator:
            return learner, *optimizers
        return self._accelerator.setup(learner, *optimizers)

    def backward(self, loss: torch.Tensor):
        """model.backward(loss)"""
        if not self._accelerator:
            loss.backward()
        else:
            self._accelerator.backward(loss)

    def eval(self):
        """Set learner to eval mode for validation"""
        self.learner.requires_grad_(False)
        self.learner.eval()

    def train(self):
        """Set learner to training mode"""
        self.learner.requires_grad_(True)
        self.learner.train()

    def save(self, path: str, save_extra: bool = False):
        """save model"""
        model = self.learner
        if save_extra:
            model = {"model": self.learner, "tracker": self.tracker}
        # TODO: save model to cloud
        with smart_open.open(path, "wb") as f:
            torch.save(model, f)
