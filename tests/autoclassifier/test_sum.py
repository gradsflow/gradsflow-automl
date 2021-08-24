
from gradsflow.autoclassifier import AutoSummarisation
from flash.text import SummarizationTask

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
    assert isinstance(model.model, SummarizationTask)
