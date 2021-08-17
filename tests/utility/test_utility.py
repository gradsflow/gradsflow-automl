import torch

from gradsflow.utility.common import create_module_index


def test_create_module_index():
    assert isinstance(create_module_index(torch.optim), dict)
