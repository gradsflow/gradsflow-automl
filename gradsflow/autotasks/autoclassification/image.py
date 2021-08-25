import torch.nn
from flash.image.classification import ImageClassifier

from gradsflow.core.autoclassifier import AutoClassifier


# noinspection PyTypeChecker
class AutoImageClassifier(AutoClassifier):
    """
    Automatically finds Image Classification Model

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
        """Build ImageClassifier model from optuna.Trial object or via keyword arguments.

        Arguments:
            backbone [str]: Image classification backbone name - resnet18, resnet50,...
            (Check Lightning-Flash for full model list)

            optimizer [str]: PyTorch Optimizers. Check `AutoImageClassification.OPTIMIZER_INDEX`
            learning_rate [float]: Learning rate for the model.
        """
        backbone = kwargs["backbone"]
        optimizer = kwargs["optimizer"]
        learning_rate = kwargs["lr"]

        return ImageClassifier(
            self.num_classes,
            backbone=backbone,
            optimizer=self.OPTIMIZER_INDEX[optimizer],
            learning_rate=learning_rate,
        )
