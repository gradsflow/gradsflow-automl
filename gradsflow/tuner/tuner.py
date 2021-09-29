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
from typing import Any, Callable, Dict, List, Union

from ray import tune
from ray.tune.sample import Domain


class ComplexObject:
    values: List[Callable]

    def __init__(self):
        self.values = []

    def __len__(self):
        return len(self.values)

    def append(self, value: Any):
        self.values.append(partial(value))

    def get_complex_object(self, idx):
        return self.values[idx]

    def to_choice(self):
        """converts to ray.tune Domain"""
        indices = tuple(range(len(self.values)))
        return tune.choice(indices)


class Tuner:
    def __init__(self):
        self._search_space: Dict[str, Domain] = {}
        self._complex_objects: Dict[str, ComplexObject] = {}

    def update_search_space(self, k: str, v: Union[Domain, ComplexObject]):
        print("received ", type(v))
        if isinstance(v, Domain):
            self._search_space[k] = v
        elif isinstance(v, ComplexObject):
            print("inside complex")
            assert isinstance(v, ComplexObject), f"Selected is_complex but object is of type {type(v)}"
            self._search_space[k] = v.to_choice()
            self._complex_objects[k] = v
        else:
            raise UserWarning(f"Tuner Search space doesn't support {type(v)}")

    def suggest_complex(self, key: str, *values: Any) -> ComplexObject:
        complex_object = ComplexObject()
        for i, v in enumerate(values):
            complex_object.append(v)

        object_choice = complex_object.to_choice()
        self._search_space[key] = object_choice
        self._complex_objects[key] = complex_object
        return complex_object

    def choice(self, key: str, *values) -> Domain:
        """Tune for categorical values"""
        x = tune.choice(values)
        self._search_space[key] = x
        return x

    def loguniform(self, key: str, lower: float, upper: float, base: float = 10) -> Domain:
        x = tune.loguniform(lower, upper, base)
        self._search_space[key] = x
        return x

    def union(self, tuner: "Tuner") -> "Tuner":
        self._search_space.update(tuner._search_space)
        self._complex_objects.update(tuner._complex_objects)
        return self

    @staticmethod
    def merge(*tuners: "Tuner"):
        tuner = Tuner()
        for _tuner in tuners:
            tuner.union(_tuner)
        return tuner

    @property
    def value(self):
        return self._search_space

    def get_complex_object(self, key: str, idx: int):
        """Get registered complex object value from key at given index"""
        print(self._complex_objects)
        return self._complex_objects[key].get_complex_object(idx)
