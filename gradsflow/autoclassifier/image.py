from typing import List, Optional, Union

import optuna
import pytorch_lightning as pl
from flash.core.data.data_module import DataModule
from flash.image.classification import ImageClassifier
from optuna.integration import PyTorchLightningPruningCallback

from gradsflow.automodel.automodel import AutoModel

DEFAULT_MODELS = ["ssl_resnet18", "ssl_resnet50"]


# noinspection PyTypeChecker
class AutoImageClassifier(AutoModel):
    def __init__(
        self,
        datamodule: DataModule,
        optimization_metric: Optional[str] = None,
        suggested_backbones: Union[List, str, None] = None,
        n_trials: int = 100,
    ):
        super().__init__(datamodule, optimization_metric, n_trials)

        if not suggested_backbones:
            self.suggested_backbones = DEFAULT_MODELS
        elif isinstance(suggested_backbones, str):
            self.suggested_backbones = [suggested_backbones]
        elif isinstance(suggested_backbones, (list, tuple)):
            self.suggested_backbones = suggested_backbones
        else:
            raise UserWarning(f"Invalid suggested_backbone type!")

        self.datamodule = datamodule
        self.num_classes = datamodule.num_classes
        self.model = None
        self.n_trials = n_trials

    # noinspection PyTypeChecker
    def build_model(self, trial: optuna.Trial):

        trial_backbone = trial.suggest_categorical("backbone", self.suggested_backbones)

        model = ImageClassifier(
            self.num_classes,
            backbone=trial_backbone,
        )

        return model

    def objective(
        self,
        trial: optuna.Trial,
    ):
        trainer = pl.Trainer(
            logger=True,
            callbacks=PyTorchLightningPruningCallback(trial, monitor="val_acc"),
        )
        model = self.build_model(trial)
        hyperparameters = dict(model=model.hparams)
        trainer.logger.log_hyperparams(hyperparameters)
        trainer.fit(model, datamodule=self.datamodule)

        return trainer.callback_metrics[self.optimization_metric].item()

    def fit(self):
        self.study.optimize(self.objective, n_trials=self.n_trials)
