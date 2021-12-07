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
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torchmetrics
from torch import nn
from torchmetrics import Metric

from gradsflow.utility.common import filter_list, module_to_cls_index

SCALAR = Union[torch.Tensor, np.float, float, int]
_nn_classes = module_to_cls_index(nn)
_tm_classes = module_to_cls_index(torchmetrics, lower_key=False)

losses: Dict[str, Callable] = {k: v for k, v in _nn_classes.items() if "loss" in k}

metrics: Dict[str, Metric] = {k: v for k, v in _tm_classes.items() if 65 <= ord(k[0]) <= 90}
metrics = {k.lower(): v for k, v in metrics.items()}


def available_losses(pattern: Optional[str] = None) -> List[str]:
    """Get available loss functions
    ```python
    >> available_losses()
    >> # crossentropy, binarycrossentropy, mae, ...

    # Filter available losses with regex pattern
    >> available_losses("m.e)
    >> # ["mae", "mse"]
    ```
    """
    loss_keys = list(losses.keys())
    return filter_list(loss_keys, pattern)


def available_metrics(pattern: Optional[str] = None) -> List[str]:
    """Get available Metrics
    ```python
    >> available_metrics()
    >> # accuracy, F1, RMSE, ...

    # Filter available metrics with regex pattern
    >> available_metrics("acc.*")
    >> # ["accuracy"]
    ```
    """
    metric_keys = list(metrics.keys())
    return filter_list(metric_keys, pattern)
