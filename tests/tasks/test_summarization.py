from unittest.mock import MagicMock

from gradsflow.autotasks import AutoSummarization


def test_build_model():
    datamodule = MagicMock()
    model = AutoSummarization(
        datamodule,
        max_epochs=1,
        timeout=5,
        suggested_backbones="sshleifer/distilbart-cnn-12-6",
        n_trials=1,
    )
    model_confs = {
        "backbone": model.DEFAULT_BACKBONES[-1],
        "optimizer": "adam",
        "lr": 1e-3,
    }
    model.build_model(**model_confs)
