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

import logging
import math
from enum import Enum
from typing import Callable, Dict, Optional

import pytorch_lightning as pl
import torch

from gradsflow.callbacks import report_checkpoint_callback
from gradsflow.core.data import AutoDataset
from gradsflow.utility.common import module_to_cls_index

logger = logging.getLogger("core.backend")


class Backend(Enum):
    # Remove torch
    pl = "pl"
    gf = "gf"
    torch = "gf"
    default = "pl"


class AutoBackend:
    _OPTIMIZER_INDEX = module_to_cls_index(torch.optim, True)

    def __init__(
        self,
        autodataset: AutoDataset,
        model_builder: Callable,
        optimization_metric: Optional[str],
        max_epochs: int = 10,
        max_steps: Optional[int] = None,
        backend: Optional[str] = None,
    ):
        self.model_builder = model_builder
        self.backend = (backend or Backend.default.value).lower()
        self.autodataset = autodataset
        self.optimization_metric = optimization_metric
        self.max_epochs = max_epochs
        self.max_steps = max_steps

    def _gf_objective(
        self,
        search_space: Dict,
        trainer_config: Dict,
        gpu: Optional[float] = 0,
    ):

        autodataset = self.autodataset
        model = self.model_builder(search_space)
        epochs = trainer_config.get("max_epochs", 1)
        tracker = model.fit(
            autodataset=autodataset,
            max_epochs=epochs,
            callbacks=trainer_config.get("callback_runner", ("tune_checkpoint", "tune_report")),
        )
        return tracker.__dict__[self.optimization_metric]

    # noinspection PyTypeChecker
    def _lightning_objective(
        self,
        config: Dict,
        trainer_config: Dict,
        gpu: Optional[float] = 0,
    ):
        val_check_interval = 1.0
        if self.max_steps:
            val_check_interval = max(self.max_steps - 1, 1.0)

        datamodule = self.autodataset.datamodule

        trainer = pl.Trainer(
            logger=True,
            checkpoint_callback=False,
            gpus=math.ceil(gpu),
            max_epochs=self.max_epochs,
            max_steps=self.max_steps,
            callbacks=[report_checkpoint_callback()],
            val_check_interval=val_check_interval,
            **trainer_config,
        )

        model = self.model_builder(config)
        hparams = dict(model=model.hparams)
        trainer.logger.log_hyperparams(hparams)
        trainer.fit(model, datamodule=datamodule)

        logger.debug(trainer.callback_metrics)
        return trainer.callback_metrics[self.optimization_metric].item()

    def optimization_objective(self, config: dict, trainer_config: dict, gpu: Optional[float] = 0.0):
        """
        Defines lightning_objective function which is used by tuner to minimize/maximize the metric.

        Args:
            config dict: key value pair of hyperparameters.
            trainer_config dict: configurations passed directly to Lightning Trainer.
            gpu Optional[float]: GPU per trial
        """
        if self.backend == Backend.pl.value:
            return self._lightning_objective(config, trainer_config, gpu)

        if self.backend in (Backend.gf.value,):
            return self._gf_objective(config, trainer_config, gpu)

        raise NotImplementedError(f"Trainer not implemented for backend: {self.backend}")
