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

from abc import abstractmethod
from typing import Dict, Optional, Union

import pytorch_lightning as pl
import torch
from flash import DataModule
from loguru import logger
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from gradsflow.utility.common import module_to_cls_index


class AutoModel:
    """
    Base model that defines hyperparameter search methods and initializes `Ray`.
    All other tasks are implementation of `AutoModel`.

    Args:
        datamodule flash.DataModule: DataModule from Flash or PyTorch Lightning
        max_epochs [int]: Maximum number of epochs for which model will train
        max_steps Optional[int]: Maximum number of steps for each epoch. Defaults None.
        optimization_metric str: Value on which hyperparameter search will run.
        By default, it is `val_accuracy`.
        n_trials int: Number of trials for HPO
        suggested_conf Dict: Any extra suggested configuration
        timeout int: HPO will stop after timeout
        prune bool: Whether to stop unpromising training.
        tune_confs Dict: raytune configurations. See more at Ray docs.
        best_trial bool: If true model will be loaded with best weights from HPO otherwise
        a best trial model without trained weights will be created.
    """

    OPTIMIZER_INDEX = module_to_cls_index(torch.optim, True)
    DEFAULT_OPTIMIZERS = ["adam", "sgd"]
    DEFAULT_LR = (1e-5, 1e-2)
    _BEST_MODEL = "best_model"
    _CURRENT_MODEL = "current_model"

    def __init__(
        self,
        datamodule: DataModule,
        max_epochs: int = 10,
        max_steps: Optional[int] = None,
        optimization_metric: Optional[str] = None,
        n_trials: int = 100,
        suggested_conf: Optional[dict] = None,
        timeout: int = 600,
        prune: bool = True,
        tune_confs: Optional[Dict] = None,
        best_trial: bool = True,
    ):
        self.analysis = None
        self.prune = prune
        self.datamodule = datamodule
        self.n_trials = n_trials
        self.best_trial = best_trial
        self.model: Union[torch.nn.Module, pl.LightningModule, None] = None
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.timeout = timeout
        self.optimization_metric = optimization_metric or "val_accuracy"
        self.optuna_confs = tune_confs or {}
        self.suggested_conf = suggested_conf or {}

        self.suggested_optimizers = self.suggested_conf.get(
            "optimizer", self.DEFAULT_OPTIMIZERS
        )
        default_lr = self.DEFAULT_LR
        self.suggested_lr = (
            self.suggested_conf.get("lr")
            or self.suggested_conf.get("learning_rate")
            or default_lr
        )

    @abstractmethod
    def _create_hparam_config(self) -> Dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def build_model(self, config: dict) -> torch.nn.Module:
        raise NotImplementedError

    # noinspection PyTypeChecker
    def objective(self, config: Dict, trainer_config: Dict):
        """
        Defines objective function which is used by tuner to minimize/maximize the metric.

        Args:
            config dict: key value pair of hyperparameters.
        """
        val_check_interval = 1.0
        if self.max_steps:
            val_check_interval = max(self.max_steps - 1, 1.0)

        datamodule = self.datamodule

        trainer = pl.Trainer(
            logger=True,
            gpus=1 if torch.cuda.is_available() else None,
            max_epochs=self.max_epochs,
            max_steps=self.max_steps,
            callbacks=[
                TuneReportCallback(
                    {
                        "val_accuracy": "val_accuracy",
                        "train_accuracy": "train_accuracy",
                    },
                    on="validation_end",
                )
            ],
            val_check_interval=val_check_interval,
            **trainer_config,
        )

        model = self.build_model(config)
        hparams = dict(model=model.hparams)
        trainer.logger.log_hyperparams(hparams)
        trainer.fit(model, datamodule=datamodule)

        logger.debug(trainer.callback_metrics)
        return trainer.callback_metrics[self.optimization_metric].item()

    def hp_tune(
        self, ray_config: Optional[dict] = None, trainer_config: Optional[dict] = None
    ):
        """
        Search Hyperparameter and builds model with the best params
        """
        trainer_config = trainer_config or {}
        ray_config = ray_config or {}

        search_space = self._create_hparam_config()
        trainable = self.objective

        analysis = tune.run(
            tune.with_parameters(trainable, trainer_config=trainer_config),
            num_samples=self.n_trials,
            metric=self.optimization_metric,
            mode="max",
            config=search_space,
            **ray_config,
        )
        self.analysis = analysis

        logger.info("🎉 Best hyperparameters found were: ", analysis.best_config)
        return analysis
