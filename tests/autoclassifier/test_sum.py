
from gradsflow.autoclassifier import AutoSummarisation

from unittest.mock import MagicMock


def test_model():
    datamodule = MagicMock()
    model = AutoSummarisation(
        datamodule,
        max_epochs=1,
        timeout=5,
        suggested_backbones="sshleifer/distilbart-cnn-12-6",
        n_trials=1,
    )
    model.hp_tune()
