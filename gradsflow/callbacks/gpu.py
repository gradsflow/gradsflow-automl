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

from loguru import logger

from gradsflow.core.callbacks import Callback
from gradsflow.utility.imports import requires


class EmissionTrackerCallback(Callback):
    """
    Tracks the carbon emissions produced by deep neural networks using
    [CodeCarbon](https://github.com/mlco2/codecarbon). To use this callback first install codecarbon using
    `pip install codecarbon`.
    For offline use, you must have to specify the [country code](https://github.com/mlco2/codecarbon#offline-mode).

    Args:
        offline: whether to use internet connection or not. You will have to provide the country code `country_iso_code` for offline use.
        **kwargs: passed directly to codecarbon class.
    """

    _name = "EmissionTrackerCallback"

    @requires("codecarbon", "install codecarbon to use EmissionTrackerCallback")
    def __init__(self, offline: bool = False, **kwargs):
        from codecarbon import EmissionsTracker, OfflineEmissionsTracker

        if offline:
            self._emission_tracker = OfflineEmissionsTracker(**kwargs)
        else:
            self._emission_tracker = EmissionsTracker(**kwargs)
        self._emission_tracker.start()

        super().__init__(model=None)

    def on_fit_end(self):
        emissions: float = self._emission_tracker.stop()
        logger.info(f"Emissions: {emissions} kg")
