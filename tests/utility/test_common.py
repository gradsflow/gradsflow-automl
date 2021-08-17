import torch

from gradsflow.utility.common import create_module_index, get_files, get_file_extension, download


def test_create_module_index():
    assert isinstance(create_module_index(torch.optim), dict)


def test_get_files():
    assert len(get_files("./")) != 0


def test_get_file_extension():
    assert get_file_extension("image.1.png") == "png"


def test_download():
    assert "gradsflow" in (download("README.md")).lower()
