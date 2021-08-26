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

import optuna
import pytorch_lightning as pl
import torch
from flash import DataModule
from loguru import logger
from optuna.integration import PyTorchLightningPruningCallback

from gradsflow.utility.common import module_to_cls_index
from gradsflow.utility.optuna import is_best_trial


class AutoModel:
    """
    Creates Optuna instance, defines methods required for hparam search

    Args:
        datamodule [flash.DataModule]: DataModule from Flash or PyTorch Lightning
        max_epochs [int]: Maximum number of epochs for which model will train
        optimization_metric [str]: Value on which hyperparameter search will run.
        By default, it is `val_accuracy`.
        n_trials [int]: Number of trials for HPO
        suggested_conf [Dict]: Any extra suggested configuration
        timeout [int]: HPO will stop after timeout
        prune [bool]: Whether to stop unpromising training.
        optuna_confs [Dict]: Optuna configs
        best_trial [bool]: If true model will be loaded with best weights from HPO otherwise
        a best trial model without trained weights will be created.
    """

    OPTIMIZER_INDEX = module_to_cls_index(torch.optim, True)
    DEFAULT_OPTIMIZERS = ["adam", "sgd"]
    DEFAULT_LR = (1e-5, 1e-1)
    _BEST_MODEL = "best_model"
    _CURRENT_MODEL = "current_model"

    def __init__(
        self,
        datamodule: DataModule,
        max_epochs: int = 10,
        optimization_metric: Optional[str] = None,
        n_trials: int = 100,
        suggested_conf: Optional[dict] = None,
        timeout: int = 600,
        prune: bool = True,
        optuna_confs: Optional[Dict] = None,
        best_trial: bool = True,
    ):

        self._pruner: optuna.pruners.BasePruner = (
            optuna.pruners.MedianPruner() if prune else optuna.pruners.NopPruner()
        )
        self.datamodule = datamodule
        self.n_trials = n_trials
        self.best_trial = best_trial
        self.model: Union[torch.nn.Module, pl.LightningModule, None] = None
        self.max_epochs = max_epochs
        self.timeout = timeout
        if not optimization_metric:
            optimization_metric = "val_accuracy"
        self.optimization_metric = optimization_metric
        if not optuna_confs:
            optuna_confs = {}
        self.optuna_confs = optuna_confs
        self._study = optuna.create_study(
            optuna_confs.get("storage"),
            pruner=self._pruner,
            study_name=optuna_confs.get("study_name"),
            direction=optuna_confs.get("direction"),
        )

        if not suggested_conf:
            suggested_conf = {}
        self.suggested_conf = suggested_conf
        self.suggested_optimizers = suggested_conf.get(
            "optimizer", self.DEFAULT_OPTIMIZERS
        )

        default_lr = self.DEFAULT_LR
        self.suggested_lr = (
            suggested_conf.get("lr")
            or suggested_conf.get("learning_rate")
            or default_lr
        )

    @abstractmethod
    def _get_trial_hparams(self, trial) -> Dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def build_model(self, **kwargs) -> torch.nn.Module:
        raise NotImplementedError

    # noinspection PyTypeChecker
    def _objective(
        self,
        trial: optuna.Trial,
    ):
        """
        Defines _objective function to minimize

        Args:
            trial [optuna.Trial]: optuna.Trial object passed during `optuna.Study.optimize`
        """
        trainer = pl.Trainer(
            logger=True,
            gpus=1 if torch.cuda.is_available() else None,
            max_epochs=self.max_epochs,
            callbacks=PyTorchLightningPruningCallback(
                trial, monitor=self.optimization_metric
            ),
        )
        trial_confs = self._get_trial_hparams(trial)
        model = self.build_model(**trial_confs)
        trial.set_user_attr(key="current_model", value=model)
        hparams = dict(model=model.hparams)
        trainer.logger.log_hyperparams(hparams)
        trainer.fit(model, datamodule=self.datamodule)

        logger.debug(trainer.callback_metrics)
        return trainer.callback_metrics[self.optimization_metric].item()

    def callback_best_trial(self, study: optuna.Study, trial: optuna.Trial) -> None:
        if is_best_trial(study, trial):
            study.set_user_attr(
                key=self._BEST_MODEL, value=trial.user_attrs[self._CURRENT_MODEL]
            )

    def hp_tune(self):
        """
        Search Hyperparameter and builds model with the best params
        """
        callbacks = []
        if self.best_trial:
            callbacks.append(self.callback_best_trial)
        self._study.optimize(
            self._objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            callbacks=callbacks,
        )

        if self.best_trial:
            self.model = self._study.user_attrs[self._BEST_MODEL]
        else:
            self.model = self.build_model(**self._study.best_params)
