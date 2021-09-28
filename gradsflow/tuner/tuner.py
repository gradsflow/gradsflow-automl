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
from dataclasses import dataclass

from ray import tune


@dataclass(init=False)
class Tuner:
    _search_space: dict

    def choice(self, key: str, *values):
        """Tune for categorical values"""
        x = tune.choice(values)
        self._search_space[key] = x
        return x

    def loguniform(self, key: str, lower: float, upper: float, base: float = 10):
        x = tune.loguniform(lower, upper, base)
        self._search_space[key] = x
        return x

    @property
    def value(self):
        return self._search_space
