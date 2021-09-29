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
import pytest
from ray.tune.sample import Domain

from gradsflow.tuner.tuner import ComplexObject, Tuner

complex_object = ComplexObject()


def test_append():
    complex_object.append("test_append")
    assert "test_append" in complex_object.values


def test_get_complex_object():
    assert complex_object.get_complex_object(0) == "test_append"


def test_to_choice():
    assert isinstance(complex_object.to_choice(), Domain)


def test_update_search_space():
    tuner = Tuner()
    tuner.update_search_space("test_update_search_space", complex_object)
    assert isinstance(tuner.get_complex_object("test_update_search_space", 0), str)

    with pytest.raises(UserWarning):
        tuner.update_search_space("hello", "world")


def test_union():
    tuner = Tuner()
    tuner1 = Tuner()
    tuner1.choice("dropout", 0.1, 0.2, 0.3)
    tuner2 = tuner.union(tuner1)
    assert tuner2.value.get("dropout") is not None


def test_merge():
    tuner1 = Tuner()
    tuner1.choice("dropout", 0.1, 0.2, 0.3)
    tuner2 = Tuner()
    tuner2.choice("layers", 1, 2, 3)
    tuner3 = Tuner.merge(tuner1, tuner2)
    assert "layers" in tuner3.value


def test_suggest_complex():
    tuner = Tuner()
    tuner.suggest_complex("test_complex", "val1", "val2")
    assert "test_complex" in tuner.value
