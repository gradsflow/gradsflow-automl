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
import math
from abc import ABC
from typing import Dict, Optional, Union

import pytorch_lightning as pl
import torch
from flash import DataModule
from loguru import logger
from ray import tune

from gradsflow.core.base import BaseAutoModel
from gradsflow.core.callbacks import report_checkpoint_callback
from gradsflow.utility.common import module_to_cls_index


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
        tune_confs Dict: raytune configurations. See more at Ray docs.
        best_trial bool: If true model will be loaded with best weights from HPO otherwise
        a best trial model without trained weights will be created.
    """

    OPTIMIZER_INDEX = module_to_cls_index(torch.optim, True)
    _DEFAULT_OPTIMIZERS = ["adam", "sgd"]
    DEFAULT_LR = (1e-5, 1e-2)

    def __init__(
        self,
        datamodule: DataModule,
        max_epochs: int = 10,
        max_steps: Optional[int] = None,
        optimization_metric: Optional[str] = None,
        n_trials: int = 20,
        suggested_conf: Optional[dict] = None,
        timeout: int = 600,
        prune: bool = True,
        tune_confs: Optional[Dict] = None,
        best_trial: bool = True,
    ):
        self.analysis = None
        self.prune = prune
        self.datamodule = datamodule
        self.n_trials = n_trials
        self.best_trial = best_trial
        self.model: Union[torch.nn.Module, pl.LightningModule, None] = None
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.timeout = timeout
        self.optimization_metric = optimization_metric or "val_accuracy"
        self.optuna_confs = tune_confs or {}
        self.suggested_conf = suggested_conf or {}

        self.suggested_optimizers = self.suggested_conf.get(
            "optimizer", self._DEFAULT_OPTIMIZERS
        )
        default_lr = self.DEFAULT_LR
        self.suggested_lr = (
            self.suggested_conf.get("lr")
            or self.suggested_conf.get("learning_rate")
            or default_lr
        )

    # noinspection PyTypeChecker
    def _objective(
        self,
        config: Dict,
        trainer_config: Dict,
        gpu: Optional[float] = 0,
    ):
        """
        Defines _objective function which is used by tuner to minimize/maximize the metric.

        Args:
            config dict: key value pair of hyperparameters.
            trainer_config dict: configurations passed directly to Lightning Trainer.
            gpu Optional[float]: GPU per trial
        """
        val_check_interval = 1.0
        if self.max_steps:
            val_check_interval = max(self.max_steps - 1, 1.0)

        datamodule = self.datamodule

        trainer = pl.Trainer(
            logger=True,
            checkpoint_callback=False,
            gpus=math.ceil(gpu),
            max_epochs=self.max_epochs,
            max_steps=self.max_steps,
            callbacks=[report_checkpoint_callback()],
            val_check_interval=val_check_interval,
            **trainer_config,
        )

        model = self.build_model(config)
        hparams = dict(model=model.hparams)
        trainer.logger.log_hyperparams(hparams)
        trainer.fit(model, datamodule=datamodule)

        logger.debug(trainer.callback_metrics)
        return trainer.callback_metrics[self.optimization_metric].item()

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
            mode Optional[str]: Whether to maximize or mimimize the `optimization_metric`.
            Values are `max` or `min`
            gpu Optional[float]: Amount of GPU resource per trial.
            cpu float: CPU cores per trial
            resume bool: Whether to resume the training or not.
        """

        trainer_config = trainer_config or {}
        ray_config = ray_config or {}

        search_space = self._create_search_space()
        trainable = self._objective

        resources_per_trial = {}
        if gpu:
            resources_per_trial["gpu"] = gpu
        if cpu:
            resources_per_trial["cpu"] = cpu

        mode = mode or "max"
        timeout_stopper = tune.stopper.TimeoutStopper(self.timeout)
        logger.info(
            "tuning hparams with metric = {} and model = {}".format(
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

        logger.info(
            "🎉 Best hyperparameters found were: {}".format(analysis.best_config)
        )
        return analysis

    def _get_best_model(self, analysis, checkpoint_file: Optional[str] = None):
        checkpoint_file = checkpoint_file or "filename"
        best_model = self.build_model(self.analysis.best_config)
        best_model = best_model.load_from_checkpoint(
            analysis.best_checkpoint + "/" + checkpoint_file
        )
        return best_model
