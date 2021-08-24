![logo](https://ik.imagekit.io/gradsflow/logo/logo-small_g2MxLWesD.png?updatedAt=1627716948296)

# An AutoML Library made with Optuna and PyTorch Lightning

[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=gradsflow_gradsflow&metric=sqale_rating)](https://sonarcloud.io/dashboard?id=gradsflow_gradsflow)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=gradsflow_gradsflow&metric=reliability_rating)](https://sonarcloud.io/dashboard?id=gradsflow_gradsflow)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=gradsflow_gradsflow&metric=security_rating)](https://sonarcloud.io/dashboard?id=gradsflow_gradsflow)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=gradsflow_gradsflow&metric=coverage)](https://sonarcloud.io/dashboard?id=gradsflow_gradsflow)
[![Documentation Status](https://readthedocs.org/projects/gradsflow/badge/?version=latest)](https://gradsflow.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/395070308.svg)](https://zenodo.org/badge/latestdoi/395070308)
[![Slack](https://img.shields.io/badge/slack-chat-orange.svg?logo=slack)](https://join.slack.com/t/gradsflow/shared_invite/zt-ulc0m0ef-xstzyowuTgYceVmFbJlBmg)

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gradsflow)](https://pypi.org/project/gradsflow/)
[![PyPI version](https://badge.fury.io/py/gradsflow.svg)](https://badge.fury.io/py/gradsflow)
[![Downloads](https://pepy.tech/badge/gradsflow)](https://pepy.tech/project/gradsflow)
[![Downloads](https://pepy.tech/badge/gradsflow/month)](https://pepy.tech/project/gradsflow)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/gradsflow/gradsflow/blob/master/LICENSE)

## Installation

#### Recommended
`pip install -U gradsflow`

#### From source
`pip install git+https://github.com/gradsflow/gradsflow@main`

## Examples

### Auto Image Classification

<details>

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

</details>

### Auto Text Classification

<details>

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

</details>

### Auto Text Summarization

<details>

```python
from gradsflow.autoclassifier import AutoSummarization

from flash.core.data.utils import download_data
from flash.text import SummarizationData

# 1. Download the data
download_data("https://pl-flash-data.s3.amazonaws.com/xsum.zip", "data/")

# 2. Load the data
datamodule = SummarizationData.from_csv(
    "input",
    "target",
    train_file="data/xsum/train.csv",
    val_file="data/xsum/valid.csv",
    test_file="data/xsum/test.csv",
)

suggested_conf = dict(
    optimizers=["adam"],
    lr=(5e-4, 1e-3),
)

model = AutoSummarization(
        datamodule,
        max_epochs=1,
        timeout=5,
        suggested_backbones="sshleifer/distilbart-cnn-12-6",
        n_trials=1,
    )

print("AutoSummarization initialised!")
model.hp_tune()
```

</details>

📑 For detailed usage examples please visit our [documentation page](https://docs.gradsflow.com).

💬 Join the [Slack](https://join.slack.com/t/gradsflow/shared_invite/zt-ulc0m0ef-xstzyowuTgYceVmFbJlBmg) group to chat with us.

## 🤗 Contribute

Contributions of any kind are welcome. Please check the [**Contributing
Guidelines**](https://github.com/gradsflow/gradsflow/blob/master/CONTRIBUTING.md) before contributing.

## Code Of Conduct

We pledge to act and interact in ways that contribute to an open, welcoming, diverse, inclusive, and healthy community.

Read full [**Contributor Covenant Code of Conduct**](https://github.com/gradsflow/gradsflow/blob/master/CODE_OF_CONDUCT.md)

## Acknowledgement

**Gradsflow** is built with help of Optuna and PyTorch Lightning 💜

## Cite
### BibTeX
```
@software{aniket_maurya_2021_5245151,
  author       = {Aniket Maurya},
  title        = {{gradsflow/gradsflow: An AutoML Library made with
                   Optuna and PyTorch Lightning}},
  month        = aug,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1b1},
  doi          = {10.5281/zenodo.5245151},
  url          = {https://doi.org/10.5281/zenodo.5245151}
}
```
