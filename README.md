# Gradsflow

An AutoML Library made with Optuna and PyTorch Lightning

[![CodeFactor](https://www.codefactor.io/repository/github/gradsflow/gradsflow/badge)](https://www.codefactor.io/repository/github/gradsflow/gradsflow)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/gradsflow/gradsflow/main.svg)](https://results.pre-commit.ci/latest/github/gradsflow/gradsflow/main)
[![Documentation Status](https://readthedocs.org/projects/gradsflow/badge/?version=latest)](https://gradsflow.readthedocs.io/en/latest/?badge=latest)


## Installation
#### Recommended
`pip install -U gradsflow`

#### From source
`pip install git+https://github.com/gradsflow/gradsflow@main`

## Examples

### Image Classification

```python
from gradsflow.autoclassifier import AutoImageClassifier

from flash.core.data.utils import download_data
from flash.image import ImageClassificationData

# 1. Create the DataModule
download_data("https://pl-flash-data.s3.amazonaws.com/hymenoptera_data.zip", "./data")

datamodule = ImageClassificationData.from_folders(
    train_folder="data/hymenoptera_data/train/",
    val_folder="data/hymenoptera_data/val/",
)

suggested_conf = dict(
    optimizers=["adam"],
    lr=(5e-4, 1e-3),
)

model = AutoImageClassifier(datamodule,
                            suggested_backbones=['ssl_resnet18'],
                            suggested_conf=suggested_conf,
                            max_epochs=1,
                            optimization_metric="val_accuracy",
                            timeout=30)

print("AutoImageClassifier initialised!")
model.hp_tune()
```


### Text Classification

```python
from gradsflow.autoclassifier import AutoTextClassifier

from flash.core.data.utils import download_data
from flash.text import TextClassificationData

# 1. Create the DataModule
download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", "./data/")

datamodule = TextClassificationData.from_csv(
    "review",
    "sentiment",
    train_file="data/imdb/train.csv",
    val_file="data/imdb/valid.csv",
    backbone="prajjwal1/bert-medium",
)

suggested_conf = dict(
    optimizers=["adam"],
    lr=(5e-4, 1e-3),
)

model = AutoTextClassifier(datamodule,
                           suggested_backbones=['sgugger/tiny-distilbert-classification'],
                           suggested_conf=suggested_conf,
                           max_epochs=1,
                           optimization_metric="val_accuracy",
                           timeout=30)

print("AutoTextClassifier initialised!")
model.hp_tune()
```
