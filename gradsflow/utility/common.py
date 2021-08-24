import inspect
import os
import sys
from glob import glob
from pathlib import Path
from typing import Any

from smart_open import open as smart_open


def download(path):
    """Read any filesystem or cloud file"""
    with smart_open(path) as fr:
        return fr.read()


def get_file_extension(path: str) -> str:
    """Returns extension of the file"""
    return os.path.basename(path).split(".")[-1]


def get_files(folder: str):
    """Fetch every file from given folder recursively."""
    folder = str(Path(folder) / "**" / "*")
    return glob(folder, recursive=True)


def module_to_cls_index(module, lower_key: bool = True) -> dict:
    """Fetch classes from module and create a Dictionary with key as class name and value as Class"""
    class_members = inspect.getmembers(sys.modules[module.__name__], inspect.isclass)
    mapping = {}
    for k, v in class_members:
        if lower_key:
            k = k.lower()
        mapping[k] = v

    return mapping


def listify(item: Any) -> list:
    """Convert any scalar value into list."""
    if not item:
        return []
    if isinstance(item, list):
        return item
    if isinstance(item, (tuple, set)):
        return list(item)
    return list(item)
