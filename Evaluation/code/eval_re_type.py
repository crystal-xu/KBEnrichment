#!/usr/bin/env python
'''
###############################################################
Usage: F1-score versus different relation types.

Run the script:
python3 eval_re_type.py --data DocRED/Dialogue
###############################################################
'''
import os.path
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


parser = argparse.ArgumentParser()
#cwd = os.getcwd()
parser.add_argument('--input_dir', type = str, default='re_data')
parser.add_argument('--output_dir', type = str, default='re_type')
parser.add_argument('--data', type=str, default='DocRED')

args = parser.parse_args()
input_dir = os.path.join(os.getcwd(), args.data, args.input_dir)
output_dir = os.path.join(os.getcwd(), args.data, args.output_dir)

# Evaluate relation types
re_type_cnt = {}
correct_re_type_cnt = {}
submission_re_type_cnt = {}

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

id2info_file = os.path.join(input_dir, 'rel_info.json')
id2rel = json.load(open(id2info_file))

output_filename = os.path.join(output_dir, 'rel_type.txt')
output_file = open(output_filename, 'w')

truth_file = os.path.join(input_dir, "test.json")
truth = json.load(open(truth_file))

std = {}
titleset = set([])

title2vectexSet = {}

list = ['', 'Id', 'Counts', 'F1-score', 'Name']
print('\t'.join(list))
output_file.write('\t'.join(list) + '\n')

re_total_cnt = 0
re_submission_cnt = 0
re_correct_cnt = 0

for x in truth:
    title = x['title']
    titleset.add(title)

    vertexSet = x['vertexSet']
    title2vectexSet[title] = vertexSet

    for label in x['labels']:
        r = label['r']
        if args.data == 'Dialogue' and r == 'unanswerable':
            continue
        h_idx = label['h']
        t_idx = label['t']
        std[(title, r, h_idx, t_idx)] = set(label['evidence'])

        if r not in re_type_cnt.keys():
            re_type_cnt[r] = 1
        else:
            re_type_cnt[r] += 1
        re_total_cnt += 1

top_re_type = []
for key, value in sorted(re_type_cnt.items(), key=lambda kv: kv[1], reverse=True):
    top_re_type.append(key)

prediction_file = os.path.join(input_dir, "dev_test_index.json")
tmp = json.load(open(prediction_file))
tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
prediction_re = [tmp[0]]
for i in range(1, len(tmp)):
    x = tmp[i]
    y = tmp[i-1]
    if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
        prediction_re.append(tmp[i])

for x in prediction_re:
    title = x['title']
    h_idx = x['h_idx']
    t_idx = x['t_idx']
    r = x['r']

    if title not in title2vectexSet:
        continue
    vertexSet = title2vectexSet[title]
    if r not in submission_re_type_cnt:
        submission_re_type_cnt[r] = 1
    else:
        submission_re_type_cnt[r] += 1
    re_submission_cnt += 1

    if (title, r, h_idx, t_idx) in std:
        if r not in correct_re_type_cnt:
            correct_re_type_cnt[r] = 1
        else:
            correct_re_type_cnt[r] += 1
        re_correct_cnt += 1

f1_re_type = {}
cnt = 1

p_total = 1.0 * re_correct_cnt / re_submission_cnt
r_total = 1.0 * re_correct_cnt / re_total_cnt
f_total = 2.0 * p_total * r_total / (p_total + r_total)
for i in range(10):
    r = top_re_type[i]
    if r not in correct_re_type_cnt:
        correct_re_type_cnt[r] = 0
    re_p_r = 1.0 * correct_re_type_cnt[r] / submission_re_type_cnt[r]
    re_r_r = 1.0 * correct_re_type_cnt[r] / re_type_cnt[r]

    if re_p_r + re_r_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p_r * re_r_r / (re_p_r + re_r_r)
    f1_re_type[r] = re_f1
    if args.data == 'DocRED':
        print('[{}] \t {} \t {} \t {:.4f} \t {}'.format(cnt, r, re_type_cnt[r], re_f1, id2rel[r]))
        output_file.write('[{}] \t {} \t {} \t {:.4f} \t {}\n'.format(cnt, r, re_type_cnt[r], re_f1, id2rel[r]))
    elif args.data == 'Dialogue':
        print('[{}] \t {} \t {} \t {:.4f} \t {}'.format(cnt, id2rel[r], re_type_cnt[r], re_f1, r))
        output_file.write('[{}] \t {} \t {} \t {:.4f} \t {}\n'.format(cnt, id2rel[r], re_type_cnt[r], re_f1, r))
    cnt += 1

plt.bar(re_type_cnt.keys(), re_type_cnt.values(), width=0.9, color='b')
print("Overall Precision={:.4f}, Recall={:.4f}, F1-score={:.4f}".format(p_total, r_total, f_total))

output_file.close()

