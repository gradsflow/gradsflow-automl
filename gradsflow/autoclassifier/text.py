from typing import List, Optional, Union

import torch
from flash.core.data.data_module import DataModule
from flash.text.classification import TextClassifier

from gradsflow.autoclassifier.base import AutoClassifier


# noinspection PyTypeChecker
class AutoTextClassifier(AutoClassifier):
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
        **kwargs
    ):
        super().__init__(
            datamodule,
            max_epochs,
            n_trials,
            optimization_metric,
            suggested_backbones,
            suggested_conf,
            **kwargs
        )

    def build_model(self, **kwargs) -> torch.nn.Module:
        backbone = kwargs["backbone"]
        optimizer = kwargs["optimizer"]
        learning_rate = kwargs["lr"]

        return TextClassifier(
            self.num_classes,
            backbone=backbone,
            optimizer=self.OPTIMIZER_INDEX[optimizer],
            learning_rate=learning_rate,
        )
