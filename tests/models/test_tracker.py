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
import pytest
from rich.table import Table

from gradsflow.models.tracker import Tracker

tracker = Tracker()


def test_reset():
    tracker.max_epochs = 5
    tracker.reset()
    assert tracker.max_epochs == 0


def test_mode():
    tracker.mode("train")
    tracker.mode("val")
    with pytest.raises(NotImplementedError):
        tracker.mode("test")


def test_track():
    tracker.track("val", 0.9, render=True)
    tracker.track("score", 0.5, render=False)


def test_create_table():
    tracker.track_loss(0.1, "train")
    tracker.track_loss(0.2, "val")
    tracker.track_metrics({"accuracy": 0.9}, mode="train")
    table = tracker.create_table()
    assert isinstance(table, Table)
