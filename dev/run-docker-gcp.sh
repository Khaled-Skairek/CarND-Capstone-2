#!/bin/bash -e

SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
PROJECT_HOME="${SCRIPT_DIR}/../"

# Copy files
# Before doing that, we have to make writable directories at `/capstone/{data,ros}`.
docker-machine scp -r ./data/ object-detection-yu:/capstone/
docker-machine scp -r ./ros/ object-detection-yu:/capstone/

# Run docker container
docker container rm carnd-capstone
docker run -it \
  -v "/capstone/ros":/capstone/ros \
  -v "/capstone/data":/capstone/data \
  -p 4567:4567 \
  --name carnd-capstone \
  carnd-capstone:latest /bin/bash
