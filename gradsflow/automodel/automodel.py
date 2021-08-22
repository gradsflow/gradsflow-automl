from abc import abstractmethod
from typing import Dict, Optional

import optuna
import pytorch_lightning as pl
import torch
from flash import DataModule
from optuna.integration import PyTorchLightningPruningCallback

from gradsflow.logging import logger
from gradsflow.utility.common import create_module_index


class AutoModel:
    OPTIMIZER_INDEX = create_module_index(torch.optim, True)

    def __init__(
        self,
        datamodule: DataModule,
        max_epochs: int = 10,
        optimization_metric: Optional[str] = None,
        n_trials: int = 100,
        suggested_conf: Optional[dict] = None,
        timeout: int = 600,
        prune: bool = True,
    ):

        self.pruner: optuna.pruners.BasePruner = (
            optuna.pruners.MedianPruner() if prune else optuna.pruners.NopPruner()
        )
        self.study = optuna.create_study(pruner=self.pruner)
        self.datamodule = datamodule
        self.n_trials = n_trials
        self.model = None
        self.max_epochs = max_epochs
        self.timeout = timeout
        if not optimization_metric:
            optimization_metric = "val_accuracy"
        self.optimization_metric = optimization_metric

        if not suggested_conf:
            suggested_conf = {}
        self.suggested_conf = suggested_conf
        self.suggested_optimizers = suggested_conf.get("optimizer", ["adam", "sgd"])

        default_lr = (1e-5, 1e-1)
        self.suggested_lr = (
            suggested_conf.get("lr")
            or suggested_conf.get("learning_rate")
            or default_lr
        )

    @abstractmethod
    def get_trial_model(self, trial) -> Dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def build_model(self, **kwargs) -> torch.nn.Module:
        raise NotImplementedError

    # noinspection PyTypeChecker
    def objective(
        self,
        trial: optuna.Trial,
    ):
        trainer = pl.Trainer(
            logger=True,
            gpus=1 if torch.cuda.is_available() else None,
            max_epochs=self.max_epochs,
            callbacks=PyTorchLightningPruningCallback(
                trial, monitor=self.optimization_metric
            ),
        )
        trial_confs = self.get_trial_model(trial)
        model = self.build_model(**trial_confs)
        hparams = dict(model=model.hparams)
        trainer.logger.log_hyperparams(hparams)
        trainer.fit(model, datamodule=self.datamodule)

        logger.info(trainer.callback_metrics)
        return trainer.callback_metrics[self.optimization_metric].item()

    def hp_tune(self):
        self.study.optimize(
            self.objective, n_trials=self.n_trials, timeout=self.timeout
        )
        self.model = self.build_model(**self.study.best_params)
