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
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import torch

from gradsflow.utility.common import AverageMeter, GDict, module_to_cls_index


class BaseAutoModel(ABC):
    """
    The main class for AutoML which consists everything required for tranining a model -
    data, model and trainer.
    """

    _OPTIMIZER_INDEX = module_to_cls_index(torch.optim, True)

    @abstractmethod
    def _create_search_space(self) -> Dict[str, str]:
        """creates search space"""
        raise NotImplementedError

    @abstractmethod
    def build_model(self, search_space: dict):
        """Build model from dictionary _search_space"""
        raise NotImplementedError


@dataclass(init=False)
class TrackingValues:
    loss: Optional[AverageMeter] = None  # Average loss in a single Epoch
    steps: Optional[int] = None  # Step per epoch
    step_loss: Optional[float] = None
    metrics: Optional[Dict[str, AverageMeter]] = None  # Average value in a single Epoch

    def __init__(self):
        self.metrics = GDict()
        self.loss = AverageMeter(name="loss")

    def update_loss(self, loss: float):
        assert isinstance(loss, (int, float, np.ndarray)), f"loss must be int | float | np.ndarray but got {type(loss)}"
        self.step_loss = loss
        self.loss.update(loss)

    def update_metrics(self, metrics: Dict[str, float]):
        """Update `TrackingValues` metrics. mode can be train or val"""
        # Track values that averages with epoch
        for key, value in metrics.items():
            try:
                self.metrics[key].update(value)
            except KeyError:
                self.metrics[key] = AverageMeter(name=key)
                self.metrics[key].update(value)

    def to_dict(self) -> dict:
        return asdict(self)

    def reset(self):
        """Values are Reset on start of each `on_*_epoch_start`"""
        self.loss.reset()
        for _, metric in self.metrics.items():
            metric.reset()
