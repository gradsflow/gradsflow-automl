import pytest
import torch
from flash.image import ImageClassificationData

from gradsflow.autoclassifier import AutoImageClassifier

datamodule = ImageClassificationData.from_folders(
    train_folder="data/hymenoptera_data/train/",
    val_folder="data/hymenoptera_data/val/",
)


def test_forward():
    model = AutoImageClassifier(datamodule)

    with pytest.raises(UserWarning):
        model.forward(torch.rand(1, 3, 8, 8))


def test_model():
    model = AutoImageClassifier(
        datamodule,
        max_epochs=1,
        timeout=5,
        suggested_backbones="ssl_resnet18",
        n_trials=1,
    )
    model.hp_tune()
