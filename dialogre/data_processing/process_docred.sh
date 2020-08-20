#!/usr/bin/env bash

for d in "train" "dev" "test"
do
    python3 convert2docredFormat.py --input_file ../data/${d}.json \
                       --output_file ../data/processed/${d}.json \

done

mv ../data/processed/train.json ../data/processed/train_annotated.json





