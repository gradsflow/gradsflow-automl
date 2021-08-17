import inspect
import os
import sys
from glob import glob
from pathlib import Path

from smart_open import open as smart_open


def download(path):
    with smart_open(path) as fr:
        return fr.read()


def get_file_extension(path: str) -> str:
    return os.path.basename(path).split(".")[-1]


def get_files(folder: str):
    folder = str(Path(folder) / "**" / "*")
    return glob(folder, recursive=True)


def create_module_index(module, lower_key: bool = True) -> dict:

    class_members = inspect.getmembers(sys.modules[module.__name__], inspect.isclass)
    mapping = {}
    for k, v in class_members:
        if lower_key:
            k = k.lower()
        mapping[k] = v

    return mapping
