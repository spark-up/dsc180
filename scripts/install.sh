#! /bin/bash

set -euo pipefail

scripts="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

get_poetry_url='https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py'

tmpdir="$(mktemp -d /tmp/replication.XXXXXX)"
CONDA_ENV="$tmpdir/venv"
export POETRY_HOME="$tmpdir/poetry"

conda create -yq -p "$CONDA_ENV" python==3.8 poetry
export "$CONDA_ENV/bin:$PATH"

poetry config virtualenvs.in-project true
poetry install -nq

bash scripts/download-zoo-resources.sh
