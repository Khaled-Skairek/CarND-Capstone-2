#!/bin/bash -e

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
PROJECT_HOME="${SCRIPT_DIR}/../"

docker run -it \
  -v "${PROJECT_HOME}/ros":/capstone/ros \
  -v "${PROJECT_HOME}/data":/capstone/data \
  --name carnd-capstone \
  carnd-capstone:latest /bin/bash
