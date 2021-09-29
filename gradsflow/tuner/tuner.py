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
from functools import partial
from typing import Any, Dict

from ray import tune
from ray.tune.sample import Domain
from torch import nn


class Tuner:
    def __init__(self):
        self._search_space: Dict[str, Domain] = {}
        self._complex_objects: Dict[str, Any] = {}

    def add_search_space(self, k: str, v: Domain):
        self._search_space[k] = v

    def suggest_complex(self, key: str, *learners: Any):
        self._complex_objects[key] = []
        for i, learner in enumerate(learners):
            self._complex_objects[key].append(partial(learner))
        learner_categorised = tuple(range(len(self._complex_objects[key])))
        self._search_space[key] = tune.choice(learner_categorised)

    def get_complex_object(self, key):
        return self._complex_objects[key]

    def choice(self, key: str, *values):
        """Tune for categorical values"""
        x = tune.choice(values)
        self._search_space[key] = x
        return x

    def loguniform(self, key: str, lower: float, upper: float, base: float = 10):
        x = tune.loguniform(lower, upper, base)
        self._search_space[key] = x
        return x

    def update(self, tuner: "Tuner"):
        self._search_space.update(tuner._search_space)
        return self

    @staticmethod
    def merge(*tuners: "Tuner"):
        tuner = Tuner()
        for _tuner in tuners:
            tuner.update(_tuner)
        return tuner

    @property
    def value(self):
        return self._search_space
