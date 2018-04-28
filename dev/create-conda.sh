#!/bin/bash -e

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
PROJECT_HOME="${SCRIPT_DIR}/.."

conda env create -y -f "${PROJECT_HOME}/environment.yml"
pip install -r "${PROJECT_HOME}/requirements.txt"
