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
import dataclasses
import os
from dataclasses import dataclass
from re import S
from typing import Any, Callable, List, Union

import torch
from accelerate import Accelerator
from torch import nn, optim
from torch.cuda import init
from torch.optim import optimizer

from gradsflow.utility.common import module_to_cls_index


@dataclass(init=False)
class Base:
    learner: Union[nn.Module, Any]
    optimizer: torch.optim.Optimizer
    loss: Union[Callable, nn._Loss]


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

    def compile(self, loss, optimizer) -> None:
        self.optimizer = self._get_optimizer(optimizer)
        self.loss = None

    def _get_optimizer(self, optimizer) -> torch.optim.Optimizer:
        if isinstance(optimizer, str):
            optimizer_fn = self._OPTIMIZER_INDEX.get(optimizer)
            assert (
                optimizer_fn is not None
            ), f"optimizer {optimizer} is not available! Available optimizers are {self._OPTIMIZER_INDEX.keys()}"
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
