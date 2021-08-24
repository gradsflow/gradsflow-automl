from abc import abstractmethod
from typing import Dict, Optional, Union

import optuna
import pytorch_lightning as pl
import torch
from flash import DataModule
from optuna.integration import PyTorchLightningPruningCallback

from gradsflow.logging import logger
from gradsflow.utility.common import module_to_cls_index


class AutoModel:
    """
    Creates Optuna instance, defines methods required for hparam search
    """

    OPTIMIZER_INDEX = module_to_cls_index(torch.optim, True)
    DEFAULT_OPTIMIZERS = ["adam", "sgd"]
    DEFAULT_LR = (1e-5, 1e-1)

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
    ):

        self._pruner: optuna.pruners.BasePruner = (
            optuna.pruners.MedianPruner() if prune else optuna.pruners.NopPruner()
        )
        self.datamodule = datamodule
        self.n_trials = n_trials
        self.model: Union[torch.nn.Module, pl.LightningModule, None] = None
        self.max_epochs = max_epochs
        self.timeout = timeout
        if not optimization_metric:
            optimization_metric = "val_accuracy"
        self.optimization_metric = optimization_metric
        if not optuna_confs:
            optuna_confs = {}
        self.optuna_confs = optuna_confs
        self.study = optuna.create_study(
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
    def _get_trial_model(self, trial) -> Dict[str, str]:
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

        Returns:

        """
        trainer = pl.Trainer(
            logger=True,
            gpus=1 if torch.cuda.is_available() else None,
            max_epochs=self.max_epochs,
            callbacks=PyTorchLightningPruningCallback(
                trial, monitor=self.optimization_metric
            ),
        )
        trial_confs = self._get_trial_model(trial)
        model = self.build_model(**trial_confs)
        hparams = dict(model=model.hparams)
        trainer.logger.log_hyperparams(hparams)
        trainer.fit(model, datamodule=self.datamodule)

        logger.debug(trainer.callback_metrics)
        return trainer.callback_metrics[self.optimization_metric].item()

    def hp_tune(self):
        """
        Search Hyperparameter and builds model with the best params
        Returns:
            sets `self.model` to the best model.

        """
        self.study.optimize(
            self._objective, n_trials=self.n_trials, timeout=self.timeout
        )
        self.model = self.build_model(**self.study.best_params)
