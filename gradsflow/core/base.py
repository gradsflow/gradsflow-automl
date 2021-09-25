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
from typing import Dict

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
        """Build model from dictionary search_space"""
        raise NotImplementedError
