#!/bin/bash -e

docker-machine create \
  --driver google \
  --google-project sage-shard-740 \
  --google-zone us-west1-b \
  --google-machine-type n1-highcpu-32 \
  object-detection-yu
