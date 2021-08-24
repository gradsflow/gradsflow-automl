from gradsflow.autoclassifier import AutoSummarization

from unittest.mock import MagicMock


def test_model():
    datamodule = MagicMock()
    model = AutoSummarization(
        datamodule,
        max_epochs=1,
        timeout=5,
        suggested_backbones="sshleifer/distilbart-cnn-12-6",
        n_trials=1,
    )
    assert model.model is None