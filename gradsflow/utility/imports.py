#  Copyright (c) 2021 GradsFlow. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import functools
import importlib
from typing import Optional


def is_installed(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except AttributeError:
        return False


def requires(package_name: str, err_msg: Optional[str] = None):
    def inner_fn(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            msg = err_msg or f"{package_name} Module must be installed to use!"
            if not is_installed(package_name):
                raise ModuleNotFoundError(msg)
            return func(*args, **kwargs)

        return wrapper

    return inner_fn
