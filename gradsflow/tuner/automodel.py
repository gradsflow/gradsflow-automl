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
import inspect
import os
from enum import Enum
from typing import Any, Dict, Optional, Union

import ray
from loguru import logger
from ray import tune
from ray.tune.sample import Domain
from torch import nn

from gradsflow import Model
from gradsflow.models.constants import LEARNER
from gradsflow.models.model import METRICS_TYPE
from gradsflow.tuner.tuner import ComplexObject, Tuner


class State(Enum):
    TUNER = "tuner"


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
        self.optimization_metric = optimization_metric or "train_loss"
        self.mode = mode or "min"
        self.analysis = None
        self._non_domain_config = {"compile": {}}

    def _register_object(self, object_name: str, variable: Any):
        if isinstance(variable, (Domain, ComplexObject)):
            self.tuner.update_search_space(object_name, variable)
            return State.TUNER
        else:
            return ray.put(variable)

    def compile(
        self,
        loss: Union[str, Domain] = None,
        optimizer: Union[str, Domain] = None,
        learning_rate: Union[float, Domain] = 3e-4,
        optimizer_config: Union[Dict[str, Domain], None] = None,
        metrics: METRICS_TYPE = None,
    ) -> None:
        compile_config = locals()
        for k, v in compile_config.items():
            logger.debug(f"compile: {k}, {v}")
            if k in ("self", "compile"):
                continue
            if isinstance(v, Domain):
                self.tuner.update_search_space(k, v)
            else:
                self._non_domain_config["compile"][k] = v

    def _get_learner(self, hparams, tuner):
        """Fetch learner from tuner or self.learner"""
        if self.learner == State.TUNER:
            idx = hparams.get(LEARNER)
            learner = tuner.get(LEARNER).get_complex_object(idx)
        else:
            learner = ray.get(self.learner)

        return learner

    def build_model(self, hparams, tuner) -> Model:
        """build and compile Model"""
        learner = self._get_learner(hparams, tuner)
        model = Model(learner)

        compile_args = set(inspect.getfullargspec(model.compile).args)
        compile_args.remove("self")
        domain_compile_args = {k: v for k, v in hparams.items() if k in compile_args}
        logger.debug(self._non_domain_config)
        logger.debug(domain_compile_args)

        model.compile(
            **self._non_domain_config["compile"],
            **domain_compile_args,
        )

        return model

    def trainable(self, search_space, autodataset, epochs, tuner):
        model: Model = self.build_model(search_space, tuner)

        model.fit(
            autodataset,
            max_epochs=epochs,
            callbacks=["tune_checkpoint", "tune_report"],
            show_progress=False,
        )

    def hp_tune(
        self,
        tuner: Tuner,
        autodataset,
        epochs: int = 1,
    ):
        self.tuner.union(tuner)
        search_space = self.tuner.value
        analysis = tune.run(
            tune.with_parameters(self.trainable, autodataset=autodataset, epochs=epochs, tuner=tuner),
            metric=self.optimization_metric,
            mode=self.mode,
            config=search_space,
        )
        self.analysis = analysis
        logger.info(f"🎉 Best HyperParameters found: {analysis.best_config}")
