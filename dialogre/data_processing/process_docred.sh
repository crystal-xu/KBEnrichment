#!/usr/bin/env bash

for d in "train" "dev" "test"
do
    python3 convert2docredFormat.py --input_file ../data/${d}.json \
                       --output_file ../data/processed/dev_${d}.json \

done





