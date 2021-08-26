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

import torch

from gradsflow.utility.common import (
    download,
    get_file_extension,
    get_files,
    listify,
    module_to_cls_index,
)


def test_create_module_index():
    assert isinstance(module_to_cls_index(torch.optim), dict)


def test_get_files():
    assert len(get_files("./")) != 0


def test_get_file_extension():
    assert get_file_extension("image.1.png") == "png"


def test_download():
    assert "gradsflow" in (download("README.md")).lower()


def test_listify():
    assert listify(None) == []
    assert listify(1) == [1]
    assert listify((1, 2)) == [1, 2]
    assert listify([1]) == [1]
    assert listify({"a": 1}) == ["a"]
