#! /bin/bash

CUDA_VISIBLE_DEVICES=1,5,6 python3 -u train.py --finetune_emb 1
