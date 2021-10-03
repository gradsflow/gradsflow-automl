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

**Recommended**

`pip install -U gradsflow`

**From Source**

`pip install git+https://github.com/gradsflow/gradsflow@main`

## Highlights

- 2021-8-25: [Released first version 0.0.1](https://pypi.org/project/gradsflow/) ✨ :tada:
- 2021-8-29: Migrated from Optuna to Ray Tune.

## What is GradsFlow?

!!! attention
    🚨 GradsFlow is changing fast. There will be a lot of breaking changes until we reach `0.1.0`.
    Feel free to give your feedback by creating an issue or join our [Slack](https://join.slack.com/t/gradsflow/shared_invite/zt-ulc0m0ef-xstzyowuTgYceVmFbJlBmg) group.

GradsFlow is an open-source AutoML Library based on PyTorch.
The goal of GradsFlow is to democratize AI and make it available to everyone.

It can automatically train Deep Learning Models for different tasks on your laptop or to a remote cluster directly from your laptop.
It also provides a powerful [Model Training API](https://docs.gradsflow.com/en/latest/gradsflow/models/model/) that can be used to train almost any PyTorch model.
GradsFlow leverages some cool OSS library including [Ray️](https://ray.io) and [PyTorch Lightning](https://https://pytorchlightning.ai/).
You don't have to write any PyTorch or hyperparameter optimization code.


- `gradsflow.core`: [Core](https://docs.gradsflow.com/en/latest/gradsflow/core/) defines the building blocks
of AutoML tasks.

- `gradsflow.autotasks`: [AutoTasks](https://docs.gradsflow.com/en/latest/gradsflow/tasks/) defines
different ML/DL tasks which is provided by gradsflow AutoML API.

- `gradsflow.model`: GradsFlow [Model](https://docs.gradsflow.com/en/latest/gradsflow/models/model/) provides a simple and
  yet customizable Model Training API.
  You can train any PyTorch model using `model.fit(...)` and it is easily customizable for more complex tasks.

- `gradsflow.tuner`: [AutoModel](https://docs.gradsflow.com/en/latest/gradsflow/tuner/) HyperParameter search with minimal code changes.


📑 Check out [notebooks examples](https://github.com/gradsflow/gradsflow/tree/main/examples/nbs) to learn more.

💙 Sponsor on [ko-fi](https://ko-fi.com/aniketmaurya)

📧 Do you need support? Contact us at <admin@gradsflow.com>


## Community

### Stay Up-to-Date
**Social**: You can also follow us on Twitter [@gradsflow](https://twitter.com/gradsflow) and [Linkedin][https://www.linkedin.com/company/gradsflow) for the latest updates.

### Questions & Discussion
💬 Join the [Slack](https://join.slack.com/t/gradsflow/shared_invite/zt-ulc0m0ef-xstzyowuTgYceVmFbJlBmg) group to chat with us.


## 🤗 Contribute

Contributions of any kind are welcome. Please check the [**Contributing
Guidelines**](https://github.com/gradsflow/gradsflow/blob/master/CONTRIBUTING.md) before contributing.

## Code Of Conduct

We pledge to act and interact in ways that contribute to an open, welcoming, diverse, inclusive, and healthy community.

Read full [**Contributor Covenant Code of Conduct**](https://github.com/gradsflow/gradsflow/blob/master/CODE_OF_CONDUCT.md)

## Acknowledgement

**GradsFlow** is built with help of awesome open-source projects (including but not limited to) PyTorch Lightning and Ray 💜

It takes inspiration from multiple APIs like [Keras](https://keras.io), [FastAI](https://docs.fast.ai).
