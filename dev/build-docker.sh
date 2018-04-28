#!/bin/bash -e

# How to make a docker machine.
# create --driver virtualbox --virtualbox-cpu-count 2  --virtualbox-memory 2048 default
#

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
PROJECT_HOME="${SCRIPT_DIR}/../"

echo "Start building a docker image."
cd "$PROJECT_HOME"  \
  && docker build --rm -t carnd-capstone:latest .
