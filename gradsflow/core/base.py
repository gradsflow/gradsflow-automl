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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch

from gradsflow.utility.common import module_to_cls_index


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


@dataclass(init=True)
class TrackingValues:
    loss: Optional[float] = None
    steps: Optional[int] = None
    step_loss: Optional[float] = None
    metrics: Union[None, Dict[str, float]] = None


@dataclass(init=False)
class BaseTracker:
    max_epochs: int = 0
    epoch: int = 0  # current train epoch
    steps_per_epoch: Optional[int] = None
    tune_metric: Optional[float] = None
    train: TrackingValues = TrackingValues()
    val: TrackingValues = TrackingValues()

    def reset(self):
        self.max_epochs = 0
        self.epoch = 0
        self.steps_per_epoch = None
        self.train = TrackingValues()
        self.val = TrackingValues()
