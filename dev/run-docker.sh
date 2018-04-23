#!/bin/bash -e

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
PROJECT_HOME="${SCRIPT_DIR}/../"

#docker run -it \
#  -v "${PROJECT_HOME}/ros":/capstone/ros \
#  -v "${PROJECT_HOME}/data":/capstone/data \
#  -p 4567:4567 \
#  --name carnd-capstone \
#  carnd-capstone:latest /bin/bash
docker run --rm -it \
  -p 4567:4567 \
  -v $PWD:/capstone \
  -v /tmp/log:/root/.ros/ \
  --name carnd-capstone \
  carnd-capstone:latest /bin/bash
