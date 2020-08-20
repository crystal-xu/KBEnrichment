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
input_dir = os.path.join(os.getcwd(), 'DocRED', args.input_dir)
output_dir = os.path.join(os.getcwd(), 'DocRED', args.output_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_filename = os.path.join(input_dir, 'test.json')
output_bert= os.path.join(output_dir, 'output_bert.json')
output_lsr= os.path.join(output_dir, 'output_lsr.json')
bert_filename = os.path.join(input_dir,'bert.json')
lsr_filename = os.path.join(input_dir, 'lsr.json')
re_data = json.load(open(input_filename))
bert_results = json.load(open(bert_filename))
lsr_results = json.load(open(lsr_filename))
std = {}
bert = {}
lsr = {}
P27 = []
P577 = []

documents = {}
for x in re_data:
    title = x['title']
    documents[title] = x
    for label in x['labels']:
        r = label['r']
        h_idx = label['h']
        t_idx = label['t']
        std[(title, h_idx, t_idx)] = (r, label['evidence'])

for x in bert_results:
    title = x['title']
    r = x['r']
    h_idx = x['h_idx']
    t_idx = x['t_idx']
    bert[(title, h_idx, t_idx)] = r

for x in lsr_results:
    title = x['title']
    r = x['r']
    h_idx = x['h_idx']
    t_idx = x['t_idx']
    lsr[(title, h_idx, t_idx)] = r

for key, value in bert.items():
    titleSet = set()
    title = key[0]
    if value == 'P27' and key in std and std[key][0] == value:
        #if key not in lsr or lsr[key] != value:
        if key not in lsr:
            print(key, value, std[key][1])
        if key[0] not in titleSet:
            titleSet.add(title)
            P27.append(documents[title])
print('-----------------------')



for key, value in lsr.items():
    titleSet = set()
    title = key[0]
    if value == 'P577' and key in std and std[key][0] == value:
        # if key not in bert or bert[key] != value:
        if key not in bert:
            print(key, value, std[key][1])
        if key[0] not in titleSet:
            titleSet.add(title)
            P577.append(documents[title])
print('-----------------------')


json.dump(P27, open(output_bert, 'w'))
json.dump(P577, open(output_lsr, 'w'))

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





