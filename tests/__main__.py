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

import urllib.request
import zipfile
from pathlib import Path

cwd = Path.cwd()
(Path.cwd() / "data").mkdir(exist_ok=True)

urllib.request.urlretrieve(
    "https://github.com/gradsflow/test-data/archive/refs/tags/cat-dog-v0.zip",
    f"{cwd}/data/test-cat-dog-v0.zip",
)

with zipfile.ZipFile(f"{cwd}/data/test-cat-dog-v0.zip", "r") as zip_ref:
    zip_ref.extractall(f"{cwd}/data/")
