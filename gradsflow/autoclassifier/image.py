import torch.nn
from flash.image.classification import ImageClassifier

from gradsflow.autoclassifier.base import AutoClassifier


# noinspection PyTypeChecker
class AutoImageClassifier(AutoClassifier):
    """
    Automatically finds Image Classification Model

    Args:
        datamodule: PL Lightning DataModule with `num_classes` property.
        max_epochs: default=10.
        n_trials: default=100.
        optimization_metric: Optional[str] = None.
        suggested_backbones: Union[List, str, None] = None.
        suggested_conf [Optional[dict] = None]: This sets Trial suggestions for optimizer,
            learning rate, and all the hyperparameters.
        timeout: Hyperparameter search will stop after timeout.

    Example:
        ```python
            suggested_conf = dict(
                optimizers=["adam"],
                lr=(5e-4, 1e-3),
            )
            model = AutoImageClassifier(datamodule,
                                        suggested_backbones=['ssl_resnet18'],
                                        suggested_conf=suggested_conf,
                                        max_epochs=1,
                                        optimization_metric="val_accuracy",
                                        timeout=30)
            model.hp_tune()
        ```
    """

    DEFAULT_BACKBONES = ["ssl_resnet18", "ssl_resnet50"]

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
