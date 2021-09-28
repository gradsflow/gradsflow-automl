import os

os.environ["GF_CI"] = "true"

import warnings
from pathlib import Path

import ray
from flash.image import ImageClassificationData

from gradsflow.data.image import image_dataset_from_directory
from gradsflow.models.model import Model
from gradsflow.tasks import autotask

warnings.filterwarnings("ignore")

ray.init(local_mode=True)

data_dir = Path.cwd()
datamodule = ImageClassificationData.from_folders(
    train_folder=f"{data_dir}/data/hymenoptera_data/train/",
    val_folder=f"{data_dir}/data/hymenoptera_data/val/",
)
data_dir = Path.cwd() / "data"

train_data = image_dataset_from_directory(f"{data_dir}/hymenoptera_data/train/", transform=True)
train_dl = train_data["dl"]

val_data = image_dataset_from_directory(f"{data_dir}/hymenoptera_data/val/", transform=True)
val_dl = val_data["dl"]


def test_build_model():
    model = autotask(
        task="image-classification",
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        num_classes=2,
        timeout=5,
        suggested_backbones="ssl_resnet18",
        n_trials=1,
    )
    kwargs = {"backbone": "ssl_resnet18", "optimizer": "adam", "lr": 1e-1}
    model.model = model.build_model(kwargs)
    assert isinstance(model.model, Model)


def test_hp_tune():
    model = autotask(
        task="image-classification",
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        num_classes=2,
        max_epochs=1,
        max_steps=2,
        timeout=30,
        suggested_backbones="ssl_resnet18",
        optimization_metric="val_accuracy",
        n_trials=1,
    )
    model.hp_tune(name="pytest-experiment", mode="max", gpu=0)
