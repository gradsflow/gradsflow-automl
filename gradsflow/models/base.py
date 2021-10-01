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
from torchmetrics import Metric, MetricCollection

from gradsflow.core.data import AutoDataset
from gradsflow.models.utils import losses
from gradsflow.models.utils import metrics as metrics_classes
from gradsflow.utility.common import module_to_cls_index


@dataclass(init=False)
class Base:
    learner: Union[nn.Module, Any]
    optimizer: torch.optim.Optimizer = None
    loss: Callable = None
    _compiled: bool = False


class BaseModel(Base):
    TEST = os.environ.get("GF_CI", "false").lower() == "true"
    _OPTIMIZER_INDEX = module_to_cls_index(torch.optim, True)

    def __init__(self, learner: Union[nn.Module, Any], accelerator_config: dict = None):
        self.accelerator = Accelerator(**accelerator_config)
        self.learner = None
        self.device = self.accelerator.device
        self.prepare_model(learner)
        self.metrics: MetricCollection = MetricCollection([])

    def prepare_model(self, learner) -> None:
        if isinstance(learner, (list, tuple)):
            self.learner = self.accelerator.prepare_model(*learner)
        elif isinstance(learner, nn.Module):
            self.learner = self.accelerator.prepare_model(learner)
        else:
            raise NotImplementedError(
                f"prepare_model is not implemented for model of type {type(learner)}! Please implement prepare_model or raise an issue."
            )

    def prepare_optimizer(self, optimizer) -> None:
        self.optimizer = self.accelerator.prepare_optimizer(optimizer)

    def prepare_data(self, autodataset: AutoDataset = None) -> AutoDataset:
        autodataset.train_dataloader = self.accelerator.prepare_data_loader(autodataset.train_dataloader)
        if autodataset.val_dataloader:
            autodataset.val_dataloader = self.accelerator.prepare_data_loader(autodataset.val_dataloader)
        return autodataset

    def add_metrics(self, *metrics: Union[str, Metric]) -> None:
        for m in metrics:
            if isinstance(m, str):
                m_cls = metrics_classes.get(m)
                assert (
                    m_cls is not None
                ), f"metrics {m} is not available! Available metrics are {tuple(metrics_classes.keys())}"
                m_obj = m_cls()
            elif isinstance(m, Metric):
                m_obj = m
            else:
                raise NotImplementedError(f"metrics not implemented for {m}! Please see `torchmetrics`.")
            self.metrics.add_metrics(m_obj)

    def _get_loss(self, loss: Union[str, Callable]) -> Optional[Callable]:
        loss_fn = None
        if isinstance(loss, str):
            loss_fn = losses.get(loss)
            assert loss_fn is not None, f"loss {loss} is not available! Available losses are {tuple(losses.keys())}"
        elif callable(loss):
            loss_fn = loss

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

    def assert_compiled(self):
        if not self._compiled:
            raise UserWarning("Model not compiled yet! Please call `model.compile(...)` first.")

    def forward(self, x):
        return self.learner(x)

    def __call__(self, x):
        return self.forward(x)

    @torch.no_grad()
    def predict(self, x):
        return self.learner(x)

    def load_from_checkpoint(self, checkpoint):
        self.learner = torch.load(checkpoint)
