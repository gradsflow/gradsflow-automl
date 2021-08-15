import os
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
