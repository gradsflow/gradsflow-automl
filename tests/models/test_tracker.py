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


def test_reset(tracker):
    tracker.max_epochs = 5
    tracker.reset()
    assert tracker.max_epochs == 0


def test_mode(tracker):
    tracker.mode("train")
    tracker.mode("val")
    with pytest.raises(NotImplementedError):
        tracker.mode("test")


def test_track(tracker):
    tracker._track("val", 0.9)
    tracker._track("score", 0.5)


def test_create_table(tracker):
    tracker.track_loss(0.1, "train")
    tracker.track_loss(0.2, "val")
    tracker.track_metrics({"accuracy": 0.9}, mode="train")
    table = tracker.create_table()
    assert isinstance(table, Table)


def test_get_item(tracker):
    assert tracker["train"] == tracker.mode("train")
    assert isinstance(tracker["metrics"], dict)
    assert "train" in tracker["loss"]
    assert "val" in tracker["loss"]
