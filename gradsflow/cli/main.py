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
import typer

from gradsflow import __version__
from gradsflow.cli import training

app = typer.Typer(
    name="GRADSFLOW CLI 🚀",
    add_completion=True,
)

app.add_typer(
    training.app,
    name="training",
)


@app.command()
def version():
    typer.echo(f"Hey 👋! You're running gradsflow version={__version__} ✨")
