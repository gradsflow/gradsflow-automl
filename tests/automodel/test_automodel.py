from flash.core.data.utils import download_data
from flash.image import ImageClassificationData

from gradsflow.automodel import AutoModel

download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "./data")

datamodule = ImageClassificationData.from_folders(
    train_folder="data/hymenoptera_data/train/",
    val_folder="data/hymenoptera_data/val/",
)


def test_auto_model():
    assert AutoModel(datamodule)
