<p align="center">
  <img width="250" alt="logo" src="https://ik.imagekit.io/gradsflow/logo/v2/gf-logo-gradsflow-orange_bv-f7gJu-up.svg"/>
  <br>
  <strong>An open-source AutoML & PyTorch Model Training Library</strong>
</p>
<p align="center">
  <a href="https://docs.gradsflow.com">Docs</a> |
  <a href="https://github.com/gradsflow/gradsflow/tree/main/examples">Examples</a>
</p>

---

[![pytest](https://github.com/gradsflow/gradsflow/actions/workflows/main.yml/badge.svg)](https://github.com/gradsflow/gradsflow/actions/workflows/main.yml)
[![Documentation Status](https://readthedocs.org/projects/gradsflow/badge/?version=latest)](https://gradsflow.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/gradsflow/gradsflow/branch/main/graph/badge.svg?token=uaB2xsf3pb)](https://codecov.io/gh/gradsflow/gradsflow)
[![PyPI version](https://badge.fury.io/py/gradsflow.svg)](https://badge.fury.io/py/gradsflow)
[![Downloads](https://pepy.tech/badge/gradsflow)](https://pepy.tech/project/gradsflow)
[![Slack](https://img.shields.io/badge/Slack-Join%20our%20community!-orange)](https://join.slack.com/t/gradsflow/shared_invite/zt-ulc0m0ef-xstzyowuTgYceVmFbJlBmg)


## Highlights

- 2021-8-25: [Released first version 0.0.1](https://pypi.org/project/gradsflow/) ✨ :tada:
- 2021-8-29: Migrated to Ray Tune
- 2021-10-4: Model Trainer support

## Installation

**Recommended**: `pip install -U gradsflow`

**Latest** (unstable): `pip install git+https://github.com/gradsflow/gradsflow@main`

## About GradsFlow

!!! attention
    🚨 GradsFlow is changing fast. There will be a lot of breaking changes until we reach `0.1.0`.
    Feel free to give your feedback by creating an issue or join our [Slack](https://join.slack.com/t/gradsflow/shared_invite/zt-ulc0m0ef-xstzyowuTgYceVmFbJlBmg) group.

GradsFlow is an open-source AutoML Library based on PyTorch.
The goal of GradsFlow is to democratize AI and make it available to everyone.

It can automatically train Deep Learning Models for different tasks on your laptop or to a remote cluster directly from your laptop.
It provides a powerful and easy-to-extend [Model Training API](https://docs.gradsflow.com/en/latest/gradsflow/models/model/)
that can be used to train almost any PyTorch model.
Though GradsFlow has its own Model Training API it also supports [PyTorch Lightning Flash](https://lightning-flash.readthedocs.io/en/latest)
to provide more rich features across different tasks.
It takes inspiration & leverages some cool OSS library including Ray️, HuggingFace Accelerate, PyTorch Lightning, Keras & fast.ai.


- `gradsflow.core`: [Core](https://docs.gradsflow.com/en/latest/gradsflow/core/) defines the building blocks
of AutoML tasks.

- `gradsflow.autotasks`: [AutoTasks](https://docs.gradsflow.com/en/latest/gradsflow/tasks/) defines
different ML/DL tasks which is provided by gradsflow AutoML API.

- `gradsflow.model`: GradsFlow [Model](https://docs.gradsflow.com/en/latest/gradsflow/models/model/) provides a simple and
  yet customizable Model Training API.
  You can train any PyTorch model using `model.fit(...)` and it is easily customizable for more complex tasks.

- `gradsflow.tuner`: [AutoModel](https://docs.gradsflow.com/en/latest/gradsflow/tuner/) HyperParameter search with minimal code changes.


📑 Check out [notebooks examples](https://github.com/gradsflow/gradsflow/tree/main/examples/nbs) to learn more.

🧡 Sponsor on [ko-fi](https://ko-fi.com/aniketmaurya)

📧 Do you need support? Contact us at <admin@gradsflow.com>


## Community

### Stay Up-to-Date
**Social**: You can also follow us on Twitter [@gradsflow](https://twitter.com/gradsflow) and [Linkedin](https://www.linkedin.com/company/gradsflow) for the latest updates.

### Questions & Discussion
💬 Join the [Slack](https://join.slack.com/t/gradsflow/shared_invite/zt-ulc0m0ef-xstzyowuTgYceVmFbJlBmg) group to chat with us.


## 🤗 Contribute

Contributions of any kind are welcome.
You can update documentation, add examples, fix identified issues, add/request a new feature.

For more details check out the [**Contributing
Guidelines**](https://github.com/gradsflow/gradsflow/blob/master/CONTRIBUTING.md) before contributing.

## Code Of Conduct

We pledge to act and interact in ways that contribute to an open, welcoming, diverse, inclusive, and healthy community.

Read full [**Contributor Covenant Code of Conduct**](https://github.com/gradsflow/gradsflow/blob/master/CODE_OF_CONDUCT.md)

## Acknowledgement

**GradsFlow** is built with help of awesome open-source projects (including but not limited to) Ray,
PyTorch Lightning, HuggingFace Accelerate, TorchMetrics.
It takes inspiration from multiple projects [Keras](https://keras.io) & [FastAI](https://docs.fast.ai).
