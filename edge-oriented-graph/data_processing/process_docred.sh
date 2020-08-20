#!/usr/bin/env bash

for d in "train" "dev" "test" "half_dev" "half_test"
do
    python3 process_docred.py --input_file ../data/DocRED/dev_${d}.json \
                       --output_file ../data/DocRED/processed/dev_${d} \
                       --data DocRED
done





