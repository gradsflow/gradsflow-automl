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
import os

import torch

from gradsflow.utility.common import module_to_cls_index


class BaseModel:
    TEST = os.environ.get("GF_CI", "false").lower() == "true"
    _OPTIMIZER_INDEX = module_to_cls_index(torch.optim, True)

    def __init__(self, model, optimizer, lr, device=None):
        self.model = model
        self.optimizer = optimizer
        self.lr = lr

        if not device:
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = "cuda"
        else:
            self.device = device

        self.model.to(self.device)

    def forward(self, x):
        return self.model(x)

    def __call__(self, x):
        return self.forward(x)

    @torch.no_grad()
    def predict(self, x):
        return self.model(x)

    def load_from_checkpoint(self, checkpoint):
        self.model = torch.load(checkpoint)
