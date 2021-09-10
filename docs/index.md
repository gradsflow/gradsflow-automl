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

GradsFlow is based on Ray and PyTorch Lightning ⚡️ (support for other torch frameworks will be added soon).
It leverages PyTorch Lightning Flash so that you don't have to write any
PyTorch or Optuna code for model building or hyperparameter tuning 🚀

Although you might want to train a custom model and search hyperparameters,
You can easily integrate any PyTorch/Lightning Flash Model with Gradsflow [AutoModel](https://docs.gradsflow.com/en/latest/gradsflow/core/) ✨

- `gradsflow.core`: [Core](https://docs.gradsflow.com/en/latest/gradsflow/core/) defines the building blocks
of AutoML tasks.

- `gradsflow.autotasks`: [AutoTasks](https://docs.gradsflow.com/en/latest/gradsflow/autotasks/) defines
different ML/DL tasks which is provided by gradsflow AutoML API.


📑 Check out [notebooks examples](https://github.com/gradsflow/gradsflow/tree/main/examples/nbs).

💬 Join the [Slack](https://join.slack.com/t/gradsflow/shared_invite/zt-ulc0m0ef-xstzyowuTgYceVmFbJlBmg) group to chat with us.

## 🤗 Contribute

Contributions of any kind are welcome. Please check the [**Contributing
Guidelines**](https://github.com/gradsflow/gradsflow/blob/master/CONTRIBUTING.md) before contributing.

## Code Of Conduct

We pledge to act and interact in ways that contribute to an open, welcoming, diverse, inclusive, and healthy community.

Read full [**Contributor Covenant Code of Conduct**](https://github.com/gradsflow/gradsflow/blob/master/CODE_OF_CONDUCT.md)

## Acknowledgement

**GradsFlow** is built with help of Ray and PyTorch Lightning 💜
