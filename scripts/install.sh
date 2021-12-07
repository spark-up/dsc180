#! /bin/bash

set -euo pipefail

scripts="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

get_poetry_url='https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py'

export POETRY_HOME="$(mktemp /tmp/poetry-home.XXXX)"
CONDA_ENV="$(mktemp /tmp/replication-env.XXXX)"

conda create -yq -p "$CONDA_ENV" python==3.8 poetry

curl "$get_poetry_url" | python - -- -yf --no-modify-path
source "$POETRY_HOME/env"

poetry config virtualenvs.in-project true
poetry install -nq

bash scripts/download-zoo-resources.sh
