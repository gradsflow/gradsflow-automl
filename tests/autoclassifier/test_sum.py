
from gradsflow.autoclassifier import AutoSummarization


def test_model():
    datamodule = MagicMock()
    model = AutoSummarization(
        datamodule,
        max_epochs=1,
        timeout=5,
        suggested_backbones="sshleifer/distilbart-cnn-12-6",
        n_trials=1,
    )
    assert isinstance(model.model, None)
