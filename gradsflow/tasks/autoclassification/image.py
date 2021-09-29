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

import timm

from gradsflow.core.autoclassifier import AutoClassifier
from gradsflow.core.backend import Backend
from gradsflow.models.model import Model


# noinspection PyTypeChecker
class AutoImageClassifier(AutoClassifier):
    """
    Automatically find Image Classification Model

    Args:
        datamodule Optional[DataModule]: PL Lightning DataModule with `num_classes` property.
        train_dataloader Optional[DataLoader]: torch dataloader
        val_dataloader Optional[DataLoader]: torch dataloader
        num_classes Optional[int]: number of classes
        max_epochs [int]: default=10.
        n_trials [int]: default=100.
        optimization_metric [Optional[str]]: defaults None
        suggested_backbones Union[List, str, None]: defaults None
        suggested_conf [Optional[dict] = None]: This sets Trial suggestions for optimizer,
            learning rate, and all the hyperparameters.
        timeout [int]: Hyperparameter search will stop after timeout.
        backend Optional[str]: Training loop code. Defaults to None.

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

            model = AutoImageClassifier(datamodule,
                                        max_epochs=10,
                                        optimization_metric="val_accuracy",
                                        timeout=300)
            model.hp_tune()
        ```
    """

    _DEFAULT_BACKBONES = ["ssl_resnet18", "ssl_resnet50"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, backend=Backend.gf.value)

    def build_model(self, config: dict) -> Model:
        """Build ImageClassifier model from `ray.tune` hyperparameter configs
        or via _search_space dictionary arguments.

        Arguments:
            backbone [str]: Image classification backbone name - resnet18, resnet50,...
            (Check Lightning-Flash for full model list)

            optimizer [str]: PyTorch Optimizers. Check `AutoImageClassification._OPTIMIZER_INDEX`
            learning_rate [float]: Learning rate for the model.
        """
        backbone = config["backbone"]

        cnn = timm.create_model(backbone, pretrained=True, num_classes=self.num_classes)
        model = Model(cnn)
        model.compile(loss="crossentropyloss", optimizer=config["optimizer"], learning_rate=config["lr"])
        return model
