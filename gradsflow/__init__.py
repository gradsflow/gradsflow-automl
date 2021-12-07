"""An open-source AutoML Library based on PyTorch"""

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

from os import environ as _environ

_environ["LOGURU_LEVEL"] = _environ.get("LOGURU_LEVEL") or _environ.get("LOG_LEVEL", "ERROR")

from gradsflow.autotasks.autoclassification.image import AutoImageClassifier
from gradsflow.autotasks.autoclassification.text import AutoTextClassifier
from gradsflow.autotasks.autosummarization import AutoSummarization
from gradsflow.autotasks.autotasks import autotask, available_tasks
from gradsflow.autotasks.engine.automodel import AutoModel
from gradsflow.data import AutoDataset
from gradsflow.models.model import Model
from gradsflow.tuner.automodel import AutoModelV2
from gradsflow.tuner.tuner import Tuner

__version__ = "0.0.8.dev0"
