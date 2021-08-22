from unittest.mock import MagicMock

from flash.core.data.utils import download_data
from flash.text import TextClassificationData

from gradsflow.autoclassifier import AutoTextClassifier


def test_build_model():
    suggested_conf = dict(
        optimizers=["adam"],
        lr=(5e-4, 1e-3),
    )
    datamodule = MagicMock()
    datamodule.num_classes = 2
    model = AutoTextClassifier(
        datamodule,
        suggested_backbones=["sgugger/tiny-distilbert-classification"],
        suggested_conf=suggested_conf,
        max_epochs=1,
        optimization_metric="val_accuracy",
        timeout=5,
        n_trials=1,
    )

    model_confs = {
        "backbone": model.DEFAULT_BACKBONES[-1],
        "optimizer": "adam",
        "lr": 1e-3,
    }
    model.build_model(**model_confs)
