#!/usr/bin/env bash

for d in "dev_train" "dev_dev" "dev_test"
do
    python3 process_dialogue.py --input_file ../data/Dialogue/dialogue_${d}.json \
                       --output_file ../data/Dialogue/processed/dialogue_${d} \
                       --data Dialogue
done





