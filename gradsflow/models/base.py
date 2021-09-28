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
from typing import Any, Callable, Optional, Union

import torch
from accelerate import Accelerator
from torch import nn

from gradsflow.models.utils import losses
from gradsflow.utility.common import module_to_cls_index


@dataclass(init=False)
class Base:
    learner: Union[nn.Module, Any]
    optimizer: torch.optim.Optimizer
    loss: Union[Callable]


class BaseModel(Base):
    TEST = os.environ.get("GF_CI", "false").lower() == "true"
    _OPTIMIZER_INDEX = module_to_cls_index(torch.optim, True)

    def __init__(self, learner: Union[nn.Module, Any], accelerator_config: dict = None):
        self.accelerator = Accelerator(**accelerator_config)
        self.learner = None
        self.device = self.accelerator.device
        self.prepare_model(learner)

    def prepare_model(self, learner) -> None:
        if isinstance(learner, (list, tuple)):
            self.learner = self.accelerator.prepare_model(*learner)
        elif isinstance(learner, nn.Module):
            self.learner = self.accelerator.prepare_model(learner)
        else:
            raise NotImplementedError(
                f"prepare_model is not implemented for model of type {type(learner)}! Please implement prepare_model or raise an issue."
            )

    def compile(self, loss, optimizer, learning_rate=3e-4, optimizer_config: Optional[dict] = None) -> None:
        optimizer_config = optimizer_config or {}
        optimizer_fn = self._get_optimizer(optimizer)
        self.optimizer = optimizer_fn(self.learner.parameters(), learning_rate=learning_rate, **optimizer_config)
        self.loss = self._get_loss(loss)

    def _get_loss(self, loss) -> Callable:
        if isinstance(loss, str):
            loss_fn = losses.get(loss)
            assert loss_fn is not None, f"loss {loss} is not available! Available losses are {tuple(losses.keys())}"

        elif callable(loss):
            loss_fn = loss
        else:
            raise NotImplementedError(f"Unknown loss {loss}")

        return loss_fn

    def _get_optimizer(self, optimizer) -> Callable:
        if isinstance(optimizer, str):
            optimizer_fn = self._OPTIMIZER_INDEX.get(optimizer)
            assert (
                optimizer_fn is not None
            ), f"optimizer {optimizer} is not available! Available optimizers are {tuple(self._OPTIMIZER_INDEX.keys())}"
        elif isinstance(optimizer, torch.optim.Optimizer):
            optimizer_fn = optimizer
        else:
            raise NotImplementedError(f"Unknown optimizer {optimizer}")
        return optimizer_fn

    def forward(self, x):
        return self.learner(x)

    def __call__(self, x):
        return self.forward(x)

    @torch.no_grad()
    def predict(self, x):
        return self.learner(x)

    def load_from_checkpoint(self, checkpoint):
        self.learner = torch.load(checkpoint)
