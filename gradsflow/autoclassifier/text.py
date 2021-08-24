import torch
from flash.text.classification import TextClassifier

from gradsflow.autoclassifier.base import AutoClassifier


# noinspection PyTypeChecker
class AutoTextClassifier(AutoClassifier):
    """
    Automatically finds Text Classification Model

    Args:
        datamodule [DataModule]: PL Lightning DataModule with `num_classes` property.
        max_epochs [int]: default=10.
        n_trials [int]: default=100.
        optimization_metric [Optional[str]]: defaults None
        suggested_backbones Union[List, str, None]: defaults None
        suggested_conf [Optional[dict] = None]: This sets Trial suggestions for optimizer,
            learning rate, and all the hyperparameters.
        timeout [int]: Hyperparameter search will stop after timeout.

    Example:
        ```python
            suggested_conf = dict(
                optimizers=["adam"],
                lr=(5e-4, 1e-3),
            )
            model = AutoTextClassifier(datamodule,
                                       suggested_backbones=['sgugger/tiny-distilbert-classification'],
                                       suggested_conf=suggested_conf,
                                       max_epochs=1,
                                       optimization_metric="val_accuracy",
                                       timeout=30)
            model.hp_tune()
        ```
    """

    DEFAULT_BACKBONES = [
        "distilbert-base-uncased-finetuned-sst-2-english",
        "sgugger/tiny-distilbert-classification",
    ]

    def build_model(self, **kwargs) -> torch.nn.Module:
        """Build ImageClassifier model from optuna.Trial object or via keyword arguments.

        kwargs:
            backbone [str]: Image classification backbone name - resnet18, resnet50,... (Check Lightning-Flash for full model list)
            optimizer [str]: PyTorch Optimizers. Check `AutoImageClassification.OPTIMIZER_INDEX`
            learning_rate [float]: Learning rate for the model.
        """
        backbone = kwargs["backbone"]
        optimizer = kwargs["optimizer"]
        learning_rate = kwargs["lr"]

        return TextClassifier(
            self.num_classes,
            backbone=backbone,
            optimizer=self.OPTIMIZER_INDEX[optimizer],
            learning_rate=learning_rate,
        )
