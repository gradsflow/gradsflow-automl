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

import logging
from abc import ABC
from typing import Optional, Union

import pytorch_lightning as pl
import torch
from ray import tune
from torch.utils.data import DataLoader

from gradsflow.core.backend import AutoBackend
from gradsflow.core.base import BaseAutoModel
from gradsflow.core.data import AutoDataset
from gradsflow.utility.common import module_to_cls_index

logger = logging.getLogger("core.model")


class AutoModel(BaseAutoModel, ABC):
    """
    Base model that defines hyperparameter search methods and initializes `Ray`.
    All other tasks are implementation of `AutoModel`.

    Args:
        datamodule flash.DataModule: DataModule from Flash or PyTorch Lightning
        max_epochs [int]: Maximum number of epochs for which model will train
        max_steps Optional[int]: Maximum number of steps for each epoch. Defaults None.
        optimization_metric str: Value on which hyperparameter search will run.
        By default, it is `val_accuracy`.
        n_trials int: Number of trials for HPO
        suggested_conf Dict: Any extra suggested configuration
        timeout int: HPO will stop after timeout
        prune bool: Whether to stop unpromising training.
        backend Optional[str]: Training backend - PL / torch / fastai. Default is PL
    """

    _OPTIMIZER_INDEX = module_to_cls_index(torch.optim, True)
    _DEFAULT_OPTIMIZERS = ["adam", "sgd"]
    _DEFAULT_LR = (1e-5, 1e-2)

    def __init__(
        self,
        datamodule: Optional[pl.LightningDataModule] = None,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
        num_classes: Optional[int] = None,
        max_epochs: int = 10,
        max_steps: Optional[int] = None,
        optimization_metric: Optional[str] = None,
        n_trials: int = 20,
        suggested_conf: Optional[dict] = None,
        timeout: int = 600,
        prune: bool = True,
        backend: Optional[str] = None,
    ):

        self.analysis = None
        self.prune = prune
        self.n_trials = n_trials
        self.model: Union[torch.nn.Module, pl.LightningModule, None] = None
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.timeout = timeout
        self.optimization_metric = optimization_metric or "val_accuracy"
        self.suggested_conf = suggested_conf or {}

        self.auto_dataset = AutoDataset(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            datamodule=datamodule,
            num_classes=num_classes,
        )

        self.datamodule = self.auto_dataset.datamodule

        self.backend = AutoBackend(
            self.auto_dataset,
            model_builder=self.build_model,
            optimization_metric=optimization_metric,
            max_epochs=max_epochs,
            max_steps=max_steps,
            backend=backend,
        )

        self.suggested_optimizers = self.suggested_conf.get("optimizer", self._DEFAULT_OPTIMIZERS)
        default_lr = self._DEFAULT_LR
        self.suggested_lr = self.suggested_conf.get("lr") or self.suggested_conf.get("learning_rate") or default_lr

    def hp_tune(
        self,
        name: Optional[str] = None,
        ray_config: Optional[dict] = None,
        trainer_config: Optional[dict] = None,
        mode: Optional[str] = None,
        gpu: Optional[float] = 0,
        cpu: Optional[float] = None,
        resume: bool = False,
    ):
        """
        Search Hyperparameter and builds model with the best params

        ```python
            automodel = AutoClassifier(data)  # implements `AutoModel`
            automodel.hp_tune(name="gflow-example", gpu=1)
        ```

        Args:
            name Optional[str]: name of the experiment.
            ray_config dict: configuration passed to `ray.tune.run(...)`
            trainer_config dict: configuration passed to `pl.trainer.fit(...)`
            mode Optional[str]: Whether to maximize or minimize the `optimization_metric`.
            Values are `max` or `min`
            gpu Optional[float]: Amount of GPU resource per trial.
            cpu float: CPU cores per trial
            resume bool: Whether to resume the training or not.
        """

        trainer_config = trainer_config or {}
        ray_config = ray_config or {}

        search_space = self._create_search_space()
        trainable = self.backend.optimization_objective

        resources_per_trial = {}
        if gpu:
            resources_per_trial["gpu"] = gpu
        if cpu:
            resources_per_trial["cpu"] = cpu

        mode = mode or "max"
        timeout_stopper = tune.stopper.TimeoutStopper(self.timeout)
        logger.info(
            "tuning _search_space with metric = {} and model = {}".format(
                self.optimization_metric,
                mode,
            )
        )
        analysis = tune.run(
            tune.with_parameters(trainable, trainer_config=trainer_config),
            name=name,
            num_samples=self.n_trials,
            metric=self.optimization_metric,
            mode=mode,
            config=search_space,
            resources_per_trial=resources_per_trial,
            resume=resume,
            stop=timeout_stopper,
            **ray_config,
        )
        self.analysis = analysis
        self.model = self._get_best_model(analysis)

        logger.info(f"🎉 Best hyperparameters found were: {analysis.best_config}")
        return analysis

    def _get_best_model(self, analysis):

        best_model = self.build_model(self.analysis.best_config)
        best_ckpt_path = analysis.best_checkpoint
        best_ckpt_path = best_ckpt_path + "/filename"
        best_model = best_model.load_from_checkpoint(best_ckpt_path)
        return best_model
