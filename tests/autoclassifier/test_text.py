from flash.core.data.utils import download_data
from flash.text import TextClassificationData

from gradsflow.autoclassifier import AutoTextClassifier

# 1. Create the DataModule
download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", "./data/")

datamodule = TextClassificationData.from_csv(
    "review",
    "sentiment",
    train_file="data/imdb/train.csv",
    val_file="data/imdb/valid.csv",
    backbone="prajjwal1/bert-medium",
)


def test_model():
    suggested_conf = dict(
        optimizers=["adam"],
        lr=(5e-4, 1e-3),
    )

    model = AutoTextClassifier(datamodule,
        suggested_backbones=['sgugger/tiny-distilbert-classification'],
        suggested_conf=suggested_conf,
        max_epochs=1,
        optimization_metric="val_accuracy",
        timeout=5,
        n_trials=1,
    )

    model.hp_tune()
