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
from unittest.mock import MagicMock, Mock, patch

import pytest

from gradsflow.core.backend import AutoBackend


@patch("gradsflow.core.backend.pl")
def test_optimization_objective(mock_pl: Mock):
    dm = MagicMock()
    model_builder = MagicMock()

    # backend is pl
    autotrainer = AutoBackend(dm, model_builder, optimization_metric="val_accuracy", backend="pl")
    autotrainer.optimization_objective({}, {})
    mock_pl.Trainer.assert_called()

    # wrong backend is passed
    with pytest.raises(NotImplementedError):
        autotrainer = AutoBackend(
            dm,
            model_builder,
            optimization_metric="val_accuracy",
            backend="error",
        )
        autotrainer.optimization_objective({}, {})
