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

from gradsflow.models import Model


class DummyModel(Model):
    def __init__(self):
        learner = torch.nn.Linear(1, 4)
        super().__init__(learner)

    def backward(self, loss: torch.Tensor):
        return None

    def train_step(self, batch):
        return {"loss": torch.as_tensor(1), "metrics": {"accuracy": 1}}

    def val_step(self, batch):
        return {"loss": torch.as_tensor(1), "metrics": {"accuracy": 1}}
