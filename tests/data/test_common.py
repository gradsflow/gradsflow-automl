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
from gradsflow.data.common import random_split_dataset
from gradsflow.data.image import get_fake_data

fake_data = get_fake_data((32, 32))


def test_random_split_dataset():
    d1, d2 = random_split_dataset(fake_data.dataset, 0.9)
    assert len(d1) > len(d2)
    assert len(d1) == int(len(fake_data.dataset) * 0.9)
