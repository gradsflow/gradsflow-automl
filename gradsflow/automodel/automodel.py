from abc import abstractmethod
from typing import Optional

import optuna
import torch
from flash import DataModule

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
    def objective(self, trial):
        raise NotImplementedError

    @abstractmethod
    def build_model(self, confs: dict):
        raise NotImplementedError

    def hp_tune(self):
        self.study.optimize(
            self.objective, n_trials=self.n_trials, timeout=self.timeout
        )
        self.model = self.build_model(**self.study.best_params)
