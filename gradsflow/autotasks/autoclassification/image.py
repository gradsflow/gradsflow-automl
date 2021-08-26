#  Copyright (c) 2021 GradsFlow. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import torch.nn
from flash.image.classification import ImageClassifier

from gradsflow.core.autoclassifier import AutoClassifier


# noinspection PyTypeChecker
class AutoImageClassifier(AutoClassifier):
    """
    Automatically find Image Classification Model

    Args:
        datamodule [DataModule]: PL Lightning DataModule with `num_classes` property.
        max_epochs [int]: default=10.
        n_trials [int]: default=100.
        optimization_metric [Optional[str]]: defaults None
        suggested_backbones Union[List, str, None]: defaults None
        suggested_conf [Optional[dict] = None]: This sets Trial suggestions for optimizer,
            learning rate, and all the hyperparameters.
        timeout [int]: Hyperparameter search will stop after timeout.

    Examples:
        ```python
            from flash.core.data.utils import download_data
            from flash.image import ImageClassificationData

            from gradsflow import AutoImageClassifier

            # 1. Create the DataModule
            download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "./data")

            datamodule = ImageClassificationData.from_folders(
                train_folder="data/hymenoptera_data/train/",
                val_folder="data/hymenoptera_data/val/",
            )

            suggested_conf = dict(
                optimizers=["adam", "sgd"],
                lr=(5e-4, 1e-3),
            )
            model = AutoImageClassifier(datamodule,
                                        suggested_conf=suggested_conf,
                                        max_epochs=10,
                                        optimization_metric="val_accuracy",
                                        timeout=300)
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
