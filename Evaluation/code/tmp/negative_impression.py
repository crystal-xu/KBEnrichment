#!/usr/bin/env python
'''
###############################################################
Usage: Get the distributions of entity-pair distances of part_of instances on DocRE train, dev and test sets.

Run the script:
python3 part_of_birth_distri.py
###############################################################
'''
import os.path
import json
import argparse

parser = argparse.ArgumentParser()
#cwd = os.getcwd()
parser.add_argument('--input_dir', type = str, default='re_data')
parser.add_argument('--output_dir', type = str, default='results')

args = parser.parse_args()
input_dir = os.path.join(os.getcwd(), 'Dialogue', args.input_dir)
output_dir = os.path.join(os.getcwd(), 'Dialogue', args.output_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_filename = os.path.join(input_dir, 'test.json')
output= os.path.join(output_dir, 'negative.json')
output_filename = os.path.join(input_dir, 'bert_lsr.json')
re_data = json.load(open(input_filename))
results = json.load(open(output_filename))
std = {}
lsr = {}
neg = []

documents = {}
for x in re_data:
    title = x['title']
    documents[title] = x
    for label in x['labels']:
        r = label['r']
        h_idx = label['h']
        t_idx = label['t']
        lsr[(title, h_idx, t_idx, r)] = 1

titleSet = set()
for x in results:
    title = x['title']
    r = x['r']
    h_idx = x['h_idx']
    t_idx = x['t_idx']
    if r == 'per:negative_impression' and (title, h_idx, t_idx, r) in lsr:
        print(title, h_idx, t_idx, r)
        if title not in titleSet:
            titleSet.add(title)
            neg.append(documents[title])


json.dump(neg, open(output, 'w'))

# for x in bert_results:
#     title = x['title']
#     r = x['r']
#     h_idx = x['h_idx']
#     t_idx = x['t_idx']
#     if (title, h_idx, t_idx) in std:
#         if r == 'P27' and std[(title, h_idx, t_idx)] == 'P27':
#             bert[(title, h_idx, t_idx)] = r
#             continue
#         if r != 'P570' and std[(title, h_idx, t_idx)] == 'P570':
#             bert[(title, h_idx, t_idx)] = r
#             continue


# for x in lsr_results:
#     title = x['title']
#     r = x['r']
#     h_idx = x['h_idx']
#     t_idx = x['t_idx']
#     if (title, h_idx, t_idx) in std:
#         if r != 'P27' and std[(title, h_idx, t_idx)] == 'P27':
#             lsr[(title, h_idx, t_idx)] = r
#             continue
#         if r == 'P570' and std[(title, h_idx, t_idx)] == 'P570':
#             lsr[(title, h_idx, t_idx)] = r
#             continue
#
# for key, value in bert.items():
#     if value == 'P27' and key in lsr_results:
#         P27.append(documents[key[0]])
#
#
# for key, value in lsr.items():
#     if value == 'P570' and key in bert_results:
#         P570.append(documents[key[0]])
#





