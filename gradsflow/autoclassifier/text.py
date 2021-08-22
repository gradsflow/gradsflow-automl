from typing import List, Optional, Union

import optuna
import pytorch_lightning as pl
import torch
from flash.core.data.data_module import DataModule
from flash.text.classification import TextClassifier
from optuna.integration import PyTorchLightningPruningCallback

from gradsflow.automodel.automodel import AutoModel
from gradsflow.logging import logger


# noinspection PyTypeChecker
class AutoTextClassifier(AutoModel):
    DEFAULT_BACKBONES = [
        "distilbert-base-uncased-finetuned-sst-2-english",
        "sgugger/tiny-distilbert-classification",
    ]

    def __init__(
        self,
        datamodule: DataModule,
        max_epochs: int = 10,
        n_trials: int = 100,
        optimization_metric: Optional[str] = None,
        suggested_backbones: Union[List, str, None] = None,
        suggested_conf: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            datamodule,
            max_epochs,
            optimization_metric,
            n_trials,
            suggested_conf,
            **kwargs,
        )

        if not suggested_backbones:
            self.suggested_backbones = self.DEFAULT_BACKBONES
        elif isinstance(suggested_backbones, str):
            self.suggested_backbones = [suggested_backbones]
        elif isinstance(suggested_backbones, (list, tuple)):
            self.suggested_backbones = suggested_backbones
        else:
            raise UserWarning(f"Invalid suggested_backbone type!")

        self.datamodule = datamodule
        self.num_classes = datamodule.num_classes

    def forward(self, x):
        if not self.model:
            raise UserWarning("model not initialized yet!")
        return self.model(x)

    # noinspection PyTypeChecker
    def get_trial_model(self, trial: optuna.Trial):

        trial_backbone = trial.suggest_categorical(
            "backbone", self.suggested_backbones)
        trial_lr = trial.suggest_float("lr", *self.suggested_lr, log=True)
        trial_optimizer = trial.suggest_categorical(
            "optimizer", self.suggested_optimizers
        )
        hparams = {
            "backbone": trial_backbone,
            "lr": trial_lr,
            "optimizer": trial_optimizer,
        }
        return hparams

    def build_model(self, **kwargs):
        backbone = kwargs["backbone"]
        optimizer = kwargs["optimizer"]
        learning_rate = kwargs["lr"]

        return TextClassifier(
            self.num_classes,
            backbone=backbone,
            optimizer=self.OPTIMIZER_INDEX[optimizer],
            learning_rate=learning_rate,
        )

    def fit(self):
        self.hp_tune()
