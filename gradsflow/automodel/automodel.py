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
        optimization_metric: Optional[str] = None,
        n_trials: int = 100,
        prune: bool = True,
    ):
        self.pruner: optuna.pruners.BasePruner = (
            optuna.pruners.MedianPruner() if prune else optuna.pruners.NopPruner()
        )
        self.study = optuna.create_study(pruner=self.pruner)
        self.datamodule = datamodule
        self.n_trials = n_trials
        self.model = None
        if not optimization_metric:
            self.optimization_metric = "val_loss"
