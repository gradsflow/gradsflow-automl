import torch
from flash.text.seq2seq import SummarizationTask

from gradsflow.core.autoclassifier import AutoClassifier


# noinspection PyTypeChecker
class AutoSummarization(AutoClassifier):
    """
    Automatically finds Text Summarization Model

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
            model = AutoSummarisation(datamodule,
                                       suggested_backbones='sshleifer/distilbart-cnn-12-6',
                                       suggested_conf=suggested_conf,
                                       max_epochs=1,
                                       optimization_metric="train_loss",
                                       timeout=30)
            model.hp_tune()
        ```
    """

    DEFAULT_BACKBONES = [
        "sshleifer/distilbart-cnn-12-6",
        "sshleifer/distilbart-xsum-12-3",
    ]

    def build_model(self, **kwargs) -> torch.nn.Module:
        """Build ImageClassifier model from optuna.Trial object or via keyword arguments.

        Arguments:
            backbone [str]: Image classification backbone name -
            sshleifer/distilbart-cnn-12-6, sshleifer/distilbart-xsum-12-3,...
            (Check Lightning-Flash for full model list)

            optimizer [str]: PyTorch Optimizers. Check `AutoImageClassification.OPTIMIZER_INDEX`
            learning_rate [float]: Learning rate for the model.

        Examples:
            ```python
                from gradsflow import AutoSummarization

                from flash.core.data.utils import download_data
                from flash.text import SummarizationData, SummarizationTask

                # 1. Download the data
                download_data("https://pl-flash-data.s3.amazonaws.com/xsum.zip", "data/")
                # 2. Load the data
                datamodule = SummarizationData.from_csv(
                    "input",
                    "target",
                    train_file="data/xsum/train.csv",
                    val_file="data/xsum/valid.csv",
                    test_file="data/xsum/test.csv",
                )

                suggested_conf = dict(
                    optimizers=["adam", "sgd"],
                    lr=(5e-4, 1e-3),
                )
                model = AutoSummarization(datamodule,
                                            suggested_conf=suggested_conf,
                                            max_epochs=10,
                                            optimization_metric="val_accuracy",
                                            timeout=300)
                model.hp_tune()
            ```

        """
        backbone = kwargs["backbone"]
        optimizer = kwargs["optimizer"]
        learning_rate = kwargs["lr"]

        return SummarizationTask(
            backbone=backbone,
            optimizer=self.OPTIMIZER_INDEX[optimizer],
            learning_rate=learning_rate,
        )
