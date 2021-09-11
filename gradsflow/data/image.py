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
import io
from typing import Callable, List, Union

from PIL import Image

from gradsflow.core.autodata import RayDataset


class ImageFolder(RayDataset):
    """Read image datasets
    ```
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png
    ```
    """

    def __init__(self, path, transforms: Union[Callable, None] = None):
        super().__init__(path)
        self.transforms = transforms
        # self.map_(lambda x: (self.file_to_class(x[0]), *x))

    @staticmethod
    def file_to_class(files: Union[str, List]):
        file_list = []
        if isinstance(files, (tuple, list)):
            for file in files:
                file_list.append(file.split("/")[-2])
            return file_list
        return files.split("/")[-2]

    def find_classes(self) -> List[str]:
        files = self.input_files
        return sorted(list(set(map(self.file_to_class, files))))

    def __iter__(self):
        for data in self.ds.iter_rows():
            x = Image.open(io.BytesIO(data[1]))
            target = self.file_to_class(data[0])
            if self.transforms:
                x = self.transforms(x)

            yield x, target
