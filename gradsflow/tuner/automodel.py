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
import os
from typing import Any, Optional, Union

from ray import tune
from ray.tune.sample import Domain
from torch import nn

from gradsflow import Model
from gradsflow.models.constants import LEARNER
from gradsflow.tuner.tuner import ComplexObject, Tuner

logger = logging.getLogger("tuner.automodel")


class AutoModelV2:
    """Searches Hyperparameter

    Examples:
        ```python
            tuner = Tuner()
            cnns = tuner.suggest_complex("learner", cnn1, cnn2)
            optimizers = tuner.choice("optimizer", "adam", "sgd")
            loss = "crossentropyloss"
            model = AutoModelV2(cnns)
            model.hp_tune(tuner, autodataset, max_epochs=10)
        ```

    Args:
        learner: tuner.
        optimization_metric: metric on which to optimize model on
        mode: max or min for optimization_metric

    """

    TEST = os.environ.get("GF_CI", "false").lower() == "true"

    def __init__(
        self,
        learner: Union[ComplexObject, nn.Module],
        optimization_metric: Optional[str] = None,
        mode: Optional[str] = None,
    ):
        self.tuner: Tuner = Tuner()
        self.learner = self._register_object(LEARNER, learner)
        self.optimization_metric = optimization_metric or "val_loss"
        self.mode = mode or "min"
        self.analysis = None

    def _register_object(self, object_name: str, variable: Any):
        if isinstance(variable, (Domain, ComplexObject)):
            self.tuner.update_search_space(object_name, variable)
        else:
            return variable

    def _build(self, hparams) -> Model:
        if self.learner is None:
            idx = hparams.get(LEARNER)
            learner = self.tuner.get_complex_object(LEARNER, idx)
        else:
            learner = self.learner
        model = Model(learner)

        model.compile(
            loss=hparams.get("loss"),
            optimizer=hparams.get("optimizer"),
            learning_rate=hparams.get("learning_rate", 1e-3),
            loss_config=hparams.get("loss_config"),
            optimizer_config=hparams.get("optimizer_config"),
        )
        return model

    def _hp_optimizer(self, search_space, autodataset, epochs):
        model = self._build(search_space)
        tracker = model.fit(
            autodataset,
            max_epochs=epochs,
            callbacks=["tune_checkpoint", "tune_report"],
        )
        return tracker.val.loss

    def hp_tune(
        self,
        tuner: Tuner,
        autodataset,
        epochs: int = 1,
    ):
        self.tuner.union(tuner)
        search_space = self.tuner.value
        analysis = tune.run(
            tune.with_parameters(self._hp_optimizer, autodataset=autodataset, epochs=epochs),
            metric=self.optimization_metric,
            mode=self.mode,
            config=search_space,
        )
        self.analysis = analysis
        logger.info(f"🎉 Best HyperParameters found: {analysis.best_config}")
