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

import torch
from flash.text.classification import TextClassifier

from gradsflow.core.autoclassifier import AutoClassifier


# noinspection PyTypeChecker
class AutoTextClassifier(AutoClassifier):
    """
    Automatically find Text Classification Model

    Examples:
        ```python
            from gradsflow import AutoTextClassifier

            from flash.core.data.utils import download_data
            from flash.text import TextClassificationData

            download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", "./data/")
            datamodule = TextClassificationData.from_csv(
                "review",
                "sentiment",
                train_file="data/imdb/train.csv",
                val_file="data/imdb/valid.csv",
            )

            model = AutoTextClassifier(datamodule,
                                       suggested_backbones=['sgugger/tiny-distilbert-classification'],
                                       max_epochs=10,
                                       optimization_metric="val_accuracy",
                                       timeout=300)
            model.hp_tune()
        ```

    Arguments:
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
    """

    _DEFAULT_BACKBONES = [
        "distilbert-base-uncased-finetuned-sst-2-english",
        "sgugger/tiny-distilbert-classification",
    ]

    def build_model(self, config: dict) -> torch.nn.Module:
        """Build TextClassifier model from `ray.tune` hyperparameter configs
        or via _search_space dictionary arguments

        Arguments:
            backbone [str]: Image classification backbone name - resnet18, resnet50,...
            (Check Lightning-Flash for full model list)

            optimizer [str]: PyTorch Optimizers. Check `AutoImageClassification._OPTIMIZER_INDEX`
            learning_rate [float]: Learning rate for the model.
        """
        self.num_classes = self.auto_dataset.num_classes
        backbone = config["backbone"]
        optimizer = config["optimizer"]
        learning_rate = config["lr"]

        return TextClassifier(
            self.num_classes,
            backbone=backbone,
            optimizer=self._OPTIMIZER_INDEX[optimizer],
            learning_rate=learning_rate,
        )
