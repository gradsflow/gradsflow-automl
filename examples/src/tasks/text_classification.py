import ray
from flash.core.data.utils import download_data
from flash.text import TextClassificationData

from gradsflow import AutoTextClassifier

ray.init(address="auto")

download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", "./data/")

print("Creating datamodule...")
datamodule = TextClassificationData.from_csv(
    "review", "sentiment", train_file="data/imdb/train.csv", val_file="data/imdb/valid.csv", batch_size=4
)

suggested_conf = dict(
    optimizers=["adam"],
    lr=(5e-4, 1e-3),
)

model = AutoTextClassifier(
    datamodule,
    suggested_backbones=["sgugger/tiny-distilbert-classification"],
    suggested_conf=suggested_conf,
    max_epochs=2,
    optimization_metric="val_accuracy",
)

print("AutoTextClassifier initialised!")
model.hp_tune()
ray.shutdown()
