from typing import Optional

import optuna
from flash import DataModule


class AutoModel:
    def __init__(
        self,
        datamodule: DataModule,
        optimization_metric: Optional[str] = None,
        n_trials: int = 100,
    ):
        self.study = optuna.create_study()
        self.datamodule = datamodule
        self.n_trials = n_trials
        if not optimization_metric:
            self.optimization_metric = "val_loss"
