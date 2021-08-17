from flash.core.data.utils import download_data
from flash.image import ImageClassificationData

from gradsflow.autoclassifier import AutoImageClassifier

# 1. Create the DataModule
download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "./data")

datamodule = ImageClassificationData.from_folders(
    train_folder="data/hymenoptera_data/train/",
    val_folder="data/hymenoptera_data/val/",
)

model = AutoImageClassifier(
    datamodule, max_epochs=2, optimization_metric="val_accuracy"
)
print("AutoImageClassifier initialised!")

model.fit()
