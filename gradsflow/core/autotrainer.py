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

import math
from enum import Enum
from typing import Callable, Dict, Optional

import pytorch_lightning as pl
import torch
from loguru import logger

from gradsflow.core.callbacks import report_checkpoint_callback
from gradsflow.utility.common import module_to_cls_index


class Backend(Enum):
    pl = "pl"
    torch = "torch"
    default = "pl"


class AutoTrainer:
    _OPTIMIZER_INDEX = module_to_cls_index(torch.optim, True)

    def __init__(
        self,
        datamodule,
        model_builder: Callable,
        optimization_metric: Optional[str],
        max_epochs: int = 10,
        max_steps: Optional[int] = None,
        backend: Optional[str] = None,
    ):
        self.model_builder = model_builder
        self.backend = (backend or Backend.default.value).lower()
        self.datamodule = datamodule
        self.optimization_metric = optimization_metric
        self.max_epochs = max_epochs
        self.max_steps = max_steps

    # def _torch_objective(self,
    #                      hparams: Dict,
    #                      trainer_config: Dict,
    #                      gpu: Optional[float] = 0, ):
    #
    #     datamodule = self.datamodule
    #     optimizer = self._OPTIMIZER_INDEX[hparams]
    #     model = self.model_builder(hparams)
    #
    #     for epoch in range(50):  # loop over the dataset multiple times
    #
    #         running_loss = 0.0
    #         for i, data in enumerate(datamodule.train_dataloader, 0):
    #             # get the inputs; data is a list of [inputs, labels]
    #             inputs, labels = data
    #
    #             # zero the parameter gradients
    #             optimizer.zero_grad()
    #
    #             # forward + backward + optimize
    #             outputs = model(inputs)
    #             loss = criterion(outputs, labels)
    #             loss.backward()
    #             optimizer.step()
    #
    #             # print statistics
    #             running_loss += loss.item()
    #             if i % 2000 == 1999:  # print every 2000 mini-batches
    #                 print('[%d, %5d] loss: %.3f' %
    #                       (epoch + 1, i + 1, running_loss / 2000))
    #                 running_loss = 0.0
    #
    #     print('Finished Training')

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

        datamodule = self.datamodule

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

    def optimization_objective(
        self, config: dict, trainer_config: dict, gpu: Optional[float] = 0.0
    ):
        """
        Defines lightning_objective function which is used by tuner to minimize/maximize the metric.

        Args:
            config dict: key value pair of hyperparameters.
            trainer_config dict: configurations passed directly to Lightning Trainer.
            gpu Optional[float]: GPU per trial
        """
        if self.backend == Backend.pl.value:
            return self._lightning_objective(config, trainer_config, gpu)

        raise NotImplementedError(
            "Trainer not implemented for backend: {}".format(self.backend)
        )
