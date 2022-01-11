#  Copyright (c) 2022 GradsFlow. All rights reserved.
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
from typing import Dict, Union

import torch
import torchmetrics
from torchmetrics import Metric, MetricCollection

from gradsflow.utility.common import module_to_cls_index

_tm_classes = module_to_cls_index(torchmetrics, lower_key=False)

metrics_classes: Dict[str, Metric] = {k: v for k, v in _tm_classes.items() if 65 <= ord(k[0]) <= 90}
metrics_classes = {k.lower(): v for k, v in metrics_classes.items()}


class MetricsContainer:
    def __init__(self, device):
        self._device = device
        self._metrics: MetricCollection = MetricCollection([])

    @property
    def metrics(self):
        return self._metrics

    def compile_metrics(self, *metrics: Union[str, Metric]) -> None:
        """Initialize metrics collection and add provided `*metrics` to the container."""
        if len(self._metrics) > 0:
            self._metrics = MetricCollection([])
        self.add_metrics(*metrics)

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
            self._metrics.add_metrics(m_obj)
        self._metrics.to(self._device)

    def _update(self, preds, target):
        """Iteratively update all the `torchmetrics` value"""
        self._metrics.update(preds, target)

    def compute(self):
        return self._metrics.compute()

    def calculate_metrics(self, preds, target) -> Dict[str, torch.Tensor]:
        """Iteratively update the compiled metrics and return the new computed values"""
        self._update(preds, target)
        return self.compute()

    def reset(self):
        """Reset the values of each of the compiled metrics"""
        self._metrics.reset()
