from flash.image import ImageClassificationData

from gradsflow.automodel import AutoModel


datamodule = ImageClassificationData.from_folders(
    train_folder="data/hymenoptera_data/train/",
    val_folder="data/hymenoptera_data/val/",
)


def test_auto_model():
    assert AutoModel(datamodule)
