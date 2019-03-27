#!/bin/bash

python main.py \
--env_name "Hopper-v2" \
--node_name "qcis5" \
--version_name "3274"

python main.py \
--env_name "Ant-v2" \
--node_name "qcis5" \
--version_name "3274"

python main.py \
--env_name "Walker2d-v2" \
--node_name "qcis5" \
--version_name "3274"

python main.py \
--env_name "Reacher-v2" \
--node_name "qcis5" \
--version_name "3274"