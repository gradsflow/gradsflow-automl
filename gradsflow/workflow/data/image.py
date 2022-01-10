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
import enum
import io

import ray
from PIL import Image
from ray import workflow

from gradsflow.data.ray_dataset import read_files


class DataFormat(enum.Enum):
    FOLDER = "from_folder"
    CSV = "from_csv"


@workflow.virtual_actor
class ImageClassificationDataWorkflow:
    def __init__(self, path: str):
        self._path = path
        self._val = 0

    @workflow.virtual_actor.readonly
    def path(self):
        return self._path

    def extract_data(self, data: str) -> ray.data.Dataset:
        data = read_files(data)
        return data

    def load_data(self, data: ray.data.Dataset) -> ray.data.Dataset:
        data = data.map(lambda x: Image.open(io.BytesIO(x[1])))
        return data

    def preprocess(self, data) -> ray.data.Dataset:
        """Preprocess dataset - eg. Image resize"""
        return data

    def flow(self):
        step = self.extract_data.step(self._path)
        step = self.load_data.step(step.run())
        step = self.preprocess.step(step)
        return step


if __name__ == "__main__":
    from pathlib import Path

    workflow.init()

    data_dir = Path.cwd()
    path = f"{data_dir}/data/test-data-cat-dog-v0/cat-dog/"
    configs = {"path": path, "data_format": DataFormat.FOLDER}
    handler = ImageClassificationDataWorkflow.get_or_create("dataflow", path)
    print(handler)

    output = handler.flow.run()
    # print(output)
    #
    for data in output.iter_rows():
        print(data)

    #
    # print(output.step(path).run())
