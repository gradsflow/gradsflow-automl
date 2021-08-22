from typing import List, Optional, Union

import torch.nn
from flash.core.data.data_module import DataModule
from flash.image.classification import ImageClassifier

from gradsflow.autoclassifier.base import AutoClassifier


# noinspection PyTypeChecker
class AutoImageClassifier(AutoClassifier):
    DEFAULT_BACKBONES = ["ssl_resnet18", "ssl_resnet50"]

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

        return ImageClassifier(
            self.num_classes,
            backbone=backbone,
            optimizer=self.OPTIMIZER_INDEX[optimizer],
            learning_rate=learning_rate,
        )
