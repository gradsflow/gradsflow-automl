![logo](https://ik.imagekit.io/gradsflow/logo/logo-small_g2MxLWesD.png?updatedAt=1627716948296)

# An open-source AutoML Library in PyTorch

[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=gradsflow_gradsflow&metric=sqale_rating)](https://sonarcloud.io/dashboard?id=gradsflow_gradsflow)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=gradsflow_gradsflow&metric=reliability_rating)](https://sonarcloud.io/dashboard?id=gradsflow_gradsflow)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=gradsflow_gradsflow&metric=security_rating)](https://sonarcloud.io/dashboard?id=gradsflow_gradsflow)
[![codecov](https://codecov.io/gh/gradsflow/gradsflow/branch/main/graph/badge.svg?token=uaB2xsf3pb)](https://codecov.io/gh/gradsflow/gradsflow)
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

## Highlights

- 2021-8-25: [Released first version 0.0.1](https://pypi.org/project/gradsflow/) ✨ :tada:
- 2021-8-29: Migrated from Optuna to Ray Tune. [Read more...](https://github.com/gradsflow/gradsflow/issues/35)

## What is GradsFlow?

!!! attention
    GradsFlow is changing fast and is not stable yet.

GradsFlow is an open-source AutoML Library for PyTorch that can train Deep Learning Models on your laptop or
to a remote cluster directly from your laptop.
Our aim is to enable non ML expert to train and build AI Products.
GradsFlow leverages the power of PyTorch Lightning ⚡️ and Ray️.
It leverages PyTorch Lightning Flash so that you don't have to write any code for model building or hyperparameter tuning 🚀

GradsFlow [Model API](https://docs.gradsflow.com/en/latest/gradsflow/model) provides a simple
[Keras like](https://keras.io) model training functionality 🔥.
You can train any PyTorch model using `model.fit(...)` and it is easily customizable for more complex tasks.

You might want to train a custom model and search hyperparameters,
You can easily integrate any PyTorch Model with Gradsflow [AutoModel](https://docs.gradsflow.com/en/latest/gradsflow/core/) ✨


- `gradsflow.core`: [Core](https://docs.gradsflow.com/en/latest/gradsflow/core/) defines the building blocks
of AutoML tasks.

- `gradsflow.autotasks`: [AutoTasks](https://docs.gradsflow.com/en/latest/gradsflow/model/) defines
different ML/DL tasks which is provided by gradsflow AutoML API.

- `gradsflow.model`: [Model](https://docs.gradsflow.com/en/latest/gradsflow/tasks/) defines the model training functionality.


📑 Check out [notebooks examples](https://github.com/gradsflow/gradsflow/tree/main/examples/nbs).

💬 Join the [Slack](https://join.slack.com/t/gradsflow/shared_invite/zt-ulc0m0ef-xstzyowuTgYceVmFbJlBmg) group to chat with us.

💙 Sponsor us on [ko-fi](https://ko-fi.com/aniketmaurya)

📧 Do you need support? Contact us at <admin@gradsflow.com>

## 🤗 Contribute

Contributions of any kind are welcome. Please check the [**Contributing
Guidelines**](https://github.com/gradsflow/gradsflow/blob/master/CONTRIBUTING.md) before contributing.

## Code Of Conduct

We pledge to act and interact in ways that contribute to an open, welcoming, diverse, inclusive, and healthy community.

Read full [**Contributor Covenant Code of Conduct**](https://github.com/gradsflow/gradsflow/blob/master/CODE_OF_CONDUCT.md)

## Acknowledgement

**GradsFlow** is built with help of awesome open-source projects (including but not limited to) PyTorch Lightning and Ray 💜

It takes inspiration from multiple APIs like Keras, FastAI.
