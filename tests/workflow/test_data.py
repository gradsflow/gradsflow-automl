#  Copyright (c) 2022 GradsFlow. All rights reserved.
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

from pathlib import Path

from gradsflow.workflow.data.image import ImageClassificationDataWorkflow

data_dir = Path.cwd()


def test_image_classification():
    path = f"{data_dir}/data/test-data-cat-dog-v0/cat-dog/"

    dataflow = ImageClassificationDataWorkflow.get_or_create("dataflow", path)
    print(dataflow)
    print(dataflow.incr.run(10))
    output = dataflow.extract_data.run(path)

    assert isinstance(next(output.iter_rows()), str)
