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
from typing import Optional, Union, List, Dict
from flash.core.data.data_module import DataModule

from .autoclassification.image import AutoImageClassifier
from .autoclassification.text import AutoTextClassifier
from .autosummarization import AutoSummarization

SUPPORTED_TASKS = {
    "image": AutoImageClassifier,
    "text": AutoTextClassifier,
    "summarsization": AutoSummarization
}


def Autotasks(
        task,
        datamodule: DataModule,
        max_epochs: int = 10,
        max_steps: int = 10,
        n_trials: int = 100,
        optimization_metric: Optional[str] = None,
        suggested_backbones: Union[List, str, None] = None,
        suggested_conf: Optional[dict] = None,
        tune_confs: Optional[Dict] = None,
        timeout: int = 600,
        prune: bool = True,
):
    """

    Args:
        task:
        datamodule [DataModule]: PL Lightning DataModule with `num_classes` property.
        max_epochs [int]: default=10.
        n_trials [int]: default=100.
        optimization_metric [Optional[str]]: defaults None
        suggested_backbones Union[List, str, None]: defaults None
        suggested_conf [Optional[dict] = None]: This sets Trial suggestions for optimizer,
            learning rate, and all the hyperparameters.
        timeout [int]: Hyperparameter search will stop after timeout.

    Returns:

    """
    if task not in SUPPORTED_TASKS:
        raise KeyError(
            "Unknown task {}, available tasks are {}".format(
                task, list(SUPPORTED_TASKS.keys())
            )
        )

    targeted_task = SUPPORTED_TASKS[task]

    return targeted_task(
        datamodule=datamodule,
        max_epochs=max_epochs,
        max_steps=max_steps,
        n_trials=n_trials,
        optimization_metric=optimization_metric,
        suggested_backbones=suggested_backbones,
        suggested_conf=suggested_conf,
        tune_confs=tune_confs,
        timeout=timeout,
        prune=prune
    )
