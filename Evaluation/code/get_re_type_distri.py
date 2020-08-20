#!/usr/bin/env python
'''
###############################################################
Usage: Get the distributions of different relation types.

Run the script:
python3 get_re_type_distri.py --data DocRED/Dialogue
###############################################################
'''
import os.path
import json
import argparse
import matplotlib.pyplot as plt
from collections import OrderedDict


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type = str, default='re_data')
parser.add_argument('--output_dir', type = str, default='re_type')
parser.add_argument('--data', type=str, default='DocRED')

args = parser.parse_args()
input_dir = os.path.join(os.getcwd(), args.data, args.input_dir)
output_dir = os.path.join(os.getcwd(), args.data, args.output_dir)

# Get the distributions of different relation types
re_type_cnt = {}

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

id2info_file = os.path.join(input_dir, 'rel_info.json')
id2rel = json.load(open(id2info_file))
rel2id = {v: k for k, v in id2rel.items()}

output_filename = os.path.join(output_dir, 're_type_distri.txt')
output_file = open(output_filename, 'w')

truth_file = os.path.join(input_dir, "train_annotated.json")
truth = json.load(open(truth_file))

std = {}
titleset = set([])

title2vectexSet = {}

re_total_cnt = 0

for x in truth:
    title = x['title']
    titleset.add(title)

    vertexSet = x['vertexSet']
    title2vectexSet[title] = vertexSet

    for label in x['labels']:
        r = label['r']
        if r == 'unanswerable':
            continue
        if r == 'P150':
            r = 'contains territorial entity'
        elif r == 'P131':
            r = 'located in territorial entity'
        elif args.data == 'DocRED':
            r = id2rel[r]

        h_idx = label['h']
        t_idx = label['t']
        std[(title, r, h_idx, t_idx)] = set(label['evidence'])

        if r not in re_type_cnt.keys():
            re_type_cnt[r] = 1
        else:
            re_type_cnt[r] += 1
        re_total_cnt += 1

top_re_type_cnt = OrderedDict()
cnt = 0
for key, value in sorted(re_type_cnt.items(), key=lambda kv: kv[1], reverse=True):
    if cnt >= 10:
        break
    top_re_type_cnt[key] = value
    cnt += 1

if args.data == 'Dialogue':
    plt.figure(figsize=(33, 30))
    plt.bar(top_re_type_cnt.keys(), top_re_type_cnt.values(), width=0.6)
    plt.xticks(rotation=50)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
else:
    plt.figure(figsize=(33, 30))
    plt.bar(top_re_type_cnt.keys(), top_re_type_cnt.values(), width=0.6)
    plt.xticks(rotation=50)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
for a, b in zip(list(top_re_type_cnt.keys()), list(top_re_type_cnt.values())):
    plt.text(a, b+0.05, '%.0f' % b, ha='center', va='bottom', fontsize=50)
plt.ylabel("Frequency", fontsize=50)
plt.savefig(os.path.join(output_dir, 're_type_distri.png'))
plt.show()

