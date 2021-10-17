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
from typing import Optional

import typer

from gradsflow.tasks.autotasks import available_tasks

app = typer.Typer()


@app.command()
def show_available_tasks():
    """
    Auto-builds Docker image for chitra ModelServer
    """
    typer.echo(available_tasks())


@app.command()
def auto_tune(
    data_path: str,
    task: str,
    max_epochs: Optional[int] = None,
    n_trials: int = 10,
    timeout: int = 600,
):
    # data = load_data()
    # data = None
    # autotask(data)
    typer.echo("to be implemented!")
