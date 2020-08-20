#!/usr/bin/env python
'''
###############################################################
Usage: F1-score versus intra- & inter- instances
for two relation types: date_of_birth and part_of.

Run the script:
python3 eval_intra_inter_given_re_type --data DocRED|Dialogue --type 'data_of_birth'|'part_of'

'P569': data_of_birth
'P361': part_of
###############################################################
'''
import os.path
import json
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type = str, default='re_data')
parser.add_argument('--output_dir', type = str, default='re_dist')
parser.add_argument('--data', type=str, default='DocRED')
parser.add_argument('--type', type=str, default='date_of_birth')
#parser.add_argument('--type', type=str, default='part_of')

args = parser.parse_args()
input_dir = os.path.join(os.getcwd(), args.data, args.input_dir)
output_dir = os.path.join(os.getcwd(), args.data, args.output_dir)
if args.type == 'date_of_birth':
    re_type = 'P569'
elif args.type == 'part_of':
    re_type = 'P361'

# Evaluate intra- v.s. inter-
re_dist_cnt = {}
correct_re_dist_cnt = {}
submission_re_dist_cnt = {}
pair_dist = {}

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_filename = os.path.join(output_dir, 'eval_date_of_birth_part_of.txt')
output_file = open(output_filename, 'w')

train_file = os.path.join(input_dir, "train_annotated.json")

truth_file = os.path.join(input_dir, "test.json")
truth = json.load(open(truth_file))

std = {}
titleset = set([])

title2vectexSet = {}

list = ['', 'Distance', 'Counts', 'Recall', 'Precision', 'F1']
print('\t'.join(list))
output_file.write('\t'.join(list) + '\n')

def find_pair_min_dist(ent1, ent2):
    dist = float('inf')
    for idx1, m1 in enumerate(ent1):
        for idx2, m2 in enumerate(ent2):
            cur_dist = abs(m1['sent_id'] - m2['sent_id'])
            dist = min(dist, cur_dist)
    return int(dist != 0)

type_data = {}

for x in truth:
    title = x['title']
    titleset.add(title)
    vertexSet = x['vertexSet']
    title2vectexSet[title] = vertexSet
    for idx1 in range(len(vertexSet)):
        for idx2 in range(len(vertexSet)):
            if idx1 == idx2:
                continue
            distance = find_pair_min_dist(vertexSet[idx1], vertexSet[idx2])
            if (title, idx1, idx2) not in pair_dist.keys():
                pair_dist[title, idx1, idx2] = distance

    for label in x['labels']:
        r = label['r']
        h_idx = label['h']
        t_idx = label['t']
        if r != re_type:
            continue
        type_data[title] = x
        distance = find_pair_min_dist(vertexSet[h_idx], vertexSet[t_idx])
        std[(title, r, h_idx, t_idx)] = distance
        if distance not in re_dist_cnt.keys():
            re_dist_cnt[distance] = 1
        else:
            re_dist_cnt[distance] += 1

top_re_dist = OrderedDict()
for key, value in sorted(re_dist_cnt.items()):
    top_re_dist[key] = value

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
    if r != re_type:
        continue
    distance = pair_dist[title, h_idx, t_idx]
    if distance not in submission_re_dist_cnt:
        submission_re_dist_cnt[distance] = 1
    else:
        submission_re_dist_cnt[distance] += 1

    if (title, re_type , h_idx, t_idx) in std and r != re_type:
        print(title, re_type, h_idx, t_idx)
        print(type_data[title])

    if (title, r, h_idx, t_idx) in std:
        distance = std[(title, r, h_idx, t_idx)]
        if distance not in correct_re_dist_cnt:
            correct_re_dist_cnt[distance] = 1
        else:
            correct_re_dist_cnt[distance] += 1

recall_re_dist = {}
precision_re_dist = {}
f1_re_dist = {}
cnt = 0
# precision, recall , f1 v.s. distance from 0-10
for distance in top_re_dist.keys():
    if distance not in correct_re_dist_cnt or distance not in re_dist_cnt or distance not in submission_re_dist_cnt:
        continue
    recall_re_dist[distance] = 1.0 * correct_re_dist_cnt[distance] / re_dist_cnt[distance]
    precision_re_dist[distance] = 1.0 * correct_re_dist_cnt[distance] / submission_re_dist_cnt[distance]
    if recall_re_dist[distance] + precision_re_dist[distance] == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * recall_re_dist[distance] * precision_re_dist[distance] / (recall_re_dist[distance] + precision_re_dist[distance])
    f1_re_dist[distance] = re_f1
    #print('\t {} \t\t {} \t\t {:.4f} \t\t {:.4f} \t\t {:.4f}'.format(distance, re_dist_cnt[distance], recall_re_dist[distance], precision_re_dist[distance], f1_re_dist[distance]))
    #output_file.write('\t {} \t\t {} \t\t {:.4f} \t\t {:.4f} \t\t {:.4f}\n'.format(distance, re_dist_cnt[distance], recall_re_dist[distance], precision_re_dist[distance], f1_re_dist[distance]))
    cnt += 1

print("---------------------------------------------")
output_file.write("---------------------------------------------\n")

def get_p_r_f1(start, end):
    correct = 0
    actual = 0
    submission = 0
    for i in range(start, end):
        if i not in re_dist_cnt or i not in submission_re_dist_cnt or i not in correct_re_dist_cnt:
            continue
        actual += re_dist_cnt[i]
        submission += submission_re_dist_cnt[i]
        correct += correct_re_dist_cnt[i]
    recall = 1.0 * correct / actual
    precision = 1.0 * correct / submission
    f1 = 2 * recall * precision / (recall + precision)
    return actual, recall, precision, f1

list = ['', 'Counts', 'Recall', 'Precision', 'F1']
# 0: Intra-
print('\t Intra-Sentence \t\t {} \t\t {:.4f} \t\t {:.4f} \t\t {:.4f}'.format(re_dist_cnt[0], recall_re_dist[0], precision_re_dist[0], f1_re_dist[0]))
output_file.write('\t Intra-Sentence \t\t {} \t\t {:.4f} \t\t {:.4f} \t\t {:.4f}\n'.format(re_dist_cnt[0], recall_re_dist[0], precision_re_dist[0], f1_re_dist[0]))

#>0: Inter-
#actual, recall, precision, f1 = get_p_r_f1(1, len(re_dist_cnt))
if 1 in re_dist_cnt and 1 in recall_re_dist and 1 in precision_re_dist and 1 in f1_re_dist:
    print('\t Inter-Sentence \t\t {} \t\t {:.4f} \t\t {:.4f} \t\t {:.4f}'.format(re_dist_cnt[1], recall_re_dist[1], precision_re_dist[1], f1_re_dist[1]))
    output_file.write('\t Inter-Sentence \t\t {} \t\t {:.4f} \t\t {:.4f} \t\t {:.4f}\n'.format(re_dist_cnt[1], recall_re_dist[1], precision_re_dist[1], f1_re_dist[1]))
output_file.close()

