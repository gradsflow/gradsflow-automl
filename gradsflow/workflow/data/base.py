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


class DataWorkflow:
    """Regular Data Workflow in Machine Learning involves `Data Extraction > Data Loading > Preprocessing > ML Training`
    Output of DataWorkflow class is a `ray_dataset.RayDataset` object.
    """

    def __init__(self):
        """Initialize important stuff"""

    def extract_data(self, data):
        """Implements Data Extraction logic"""

    def load_data(self, data):
        """Implements Data Loading logic for each data point in the pool of extracted data."""

    def preprocess(self, data):
        """Preprocess dataset - eg. Image resize"""

    def flow(self):
        """
        step = self.extract_data.step(self._path, name="extract_data")
        step = self.load_data.step(step, name="load_data")
        step = self.preprocess.step(step, name="preprocess")
        """
