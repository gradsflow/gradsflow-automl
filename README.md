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

<!--- BADGES: START --->
[![GitHub - License](https://img.shields.io/github/license/gradsflow/gradsflow?logo=github&style=flat&color=green)][#github-license]
[![pytest](https://github.com/gradsflow/gradsflow/actions/workflows/main.yml/badge.svg)][#pytest-package]
[![Documentation Status](https://readthedocs.org/projects/gradsflow/badge/?version=latest)](https://gradsflow.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/gradsflow/gradsflow/branch/main/graph/badge.svg?token=uaB2xsf3pb)](https://codecov.io/gh/gradsflow/gradsflow)
[![PyPI version](https://badge.fury.io/py/gradsflow.svg)](https://badge.fury.io/py/gradsflow)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gradsflow?logo=pypi&style=flat&color=blue)][#pypi-package]
[![Downloads](https://pepy.tech/badge/gradsflow)](https://pepy.tech/project/gradsflow)
[![Conda - Platform](https://img.shields.io/conda/pn/conda-forge/gradsflow?logo=anaconda&style=flat)][#conda-forge-package]
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/gradsflow?logo=anaconda&style=flat&color=orange)][#conda-forge-package]
[![Slack](https://img.shields.io/badge/Slack-Join%20our%20community!-orange)][#slack-package]

[#github-license]: https://github.com/gradsflow/gradsflow/blob/main/LICENSE
[#pypi-package]: https://pypi.org/project/gradsflow/
[#conda-forge-package]: https://anaconda.org/conda-forge/gradsflow
[#slack-package]: https://join.slack.com/t/gradsflow/shared_invite/zt-ulc0m0ef-xstzyowuTgYceVmFbJlBmg
[#pytest-package]: https://github.com/gradsflow/gradsflow/actions/workflows/main.yml
<!--- BADGES: END --->

## Highlights
- 2021-10-7: [v0.0.6 Release blog post](https://towardsdatascience.com/gradsflow-democratizing-ai-with-automl-9a8a75d6b7ea)
- 2021-10-5: [Hacktoberfest 2021 Kickoff event](https://youtu.be/lVtxXyCAZ-4?t=2647)
- 2021-10-4: Model Trainer support
- 2021-8-29: Migrated to Ray Tune
- 2021-8-25: [Released first version 0.0.1](https://pypi.org/project/gradsflow/) ✨ :tada:

## About GradsFlow

!!! attention
    🚨 GradsFlow is changing fast. There will be a lot of breaking changes until we reach `0.1.0`.
    Feel free to give your feedback by creating an issue or join our [Slack](https://join.slack.com/t/gradsflow/shared_invite/zt-ulc0m0ef-xstzyowuTgYceVmFbJlBmg) group.

GradsFlow is an open-source AutoML Library based on PyTorch.
Our goal is to democratize AI and make it available to everyone.

It can automatically build & train Deep Learning Models for different tasks on your laptop or to a remote cluster
directly from your laptop.
It provides a powerful and easy-to-extend [Model Training API](https://docs.gradsflow.com/en/latest/gradsflow/models/model/)
that can be used to train almost any PyTorch model.
Though GradsFlow has its own Model Training API it also supports [PyTorch Lightning Flash](https://lightning-flash.readthedocs.io/en/latest)
to provide more rich features across different tasks.


!!! info
    Gradsflow is built for both *beginners* and *experts*! `AutoTasks` provides zero-code AutoML while
    `Model` and `Tuner` provides custom model training and Hyperparameter optimization.


### Installation

**Recommended**:

The recommended method of installing `gradsflow` is either with `pip` from PyPI or, with `conda` from conda-forge channel.

- **with pip**

  ```sh
  pip install -U gradsflow
  ```

- **with conda**

  ```sh
  conda install -c conda-forge gradsflow
  ```

**Latest** (unstable):

You can also install the latest bleeding edge version (could be unstable) of `gradsflow`, should you feel motivated enough, as follows:

```sh
pip install git+https://github.com/gradsflow/gradsflow@main
```

### Automatic Model Building and Training
Are you a beginner or from non Machine Learning background? This section is for you. Gradsflow `AutoTask` provides
automatic model building and training across various different tasks
including Image Recognition, Sentiment Analysis, Text Summarization and more to come.

![autotextsummarization](https://ik.imagekit.io/gradsflow/docs/gf/autotextsummarization_9vRXj5mWG9P)


### Simplified Hyperparameter tuning API
`Tuner` provides a simplified API to move from Model Training to Hyperparameter optimization.

![model training image](https://ik.imagekit.io/gradsflow/docs/gf/gradsflow-model-training_B1HZpLFRv8.png)


### Components

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
