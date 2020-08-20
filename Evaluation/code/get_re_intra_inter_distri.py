#!/usr/bin/env python
'''
###############################################################
Usage: Get the distributions of intra- v.s. inter-sentential instances.

Run the script:
python3 get_re_intra_inter_distri.py --data DocRED/Dialogue
###############################################################
'''
import os.path
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type = str, default='re_data')
parser.add_argument('--output_dir', type = str, default='re_dist')
parser.add_argument('--data', type=str, default='DocRED')

args = parser.parse_args()
input_dir = os.path.join(os.getcwd(), args.data, args.input_dir)
output_dir = os.path.join(os.getcwd(), args.data, args.output_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_filename = os.path.join(output_dir, 're_intra_inter_distri.txt')
output_file = open(output_filename, 'w')

list = ['', 'Counts', 'Precentage']
print('\t'.join(list))
output_file.write('\t'.join(list) + '\n')

def find_pair_min_dist(ent1, ent2):
    dist = float('inf')
    for idx1, m1 in enumerate(ent1):
        for idx2, m2 in enumerate(ent2):
            cur_dist = abs(m1['sent_id'] - m2['sent_id'])
            dist = min(dist, cur_dist)
    return dist

def inter_intra_cnt(data_file_name):
    intra_cnt = 0
    inter_cnt = 0
    data_file_name = os.path.join(input_dir, data_file_name)
    truth = json.load(open(data_file_name))
    for x in truth:
        vertexSet = x['vertexSet']

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            distance = find_pair_min_dist(vertexSet[h_idx], vertexSet[t_idx])
            if distance == 0:
                intra_cnt += 1
            else:
                inter_cnt += 1
    return intra_cnt, inter_cnt

intra_cnt, inter_cnt = inter_intra_cnt('train_annotated.json')
print("training set")
output_file.write("training set\n")
print('\t Intra-Sentence \t\t {} \t\t {:.4f}'.format(intra_cnt, intra_cnt/(intra_cnt + inter_cnt)*100))
print('\t Inter-Sentence \t\t {} \t\t {:.4f}'.format(inter_cnt, inter_cnt/(intra_cnt+inter_cnt)*100))
output_file.write('\t Intra-Sentence \t\t {} \t\t {:.4f}'.format(intra_cnt, intra_cnt/(intra_cnt + inter_cnt)*100))
output_file.write('\t Inter-Sentence \t\t {} \t\t {:.4f}'.format(inter_cnt, inter_cnt/(intra_cnt+inter_cnt)*100))
print("total: ", intra_cnt + inter_cnt)
print("dev set")
output_file.write("dev* set\n")
intra_cnt, inter_cnt = inter_intra_cnt('dev.json')
print('\t Intra-Sentence \t\t {} \t\t {:.4f}'.format(intra_cnt, intra_cnt/(intra_cnt + inter_cnt)*100))
print('\t Inter-Sentence \t\t {} \t\t {:.4f}'.format(inter_cnt, inter_cnt/(intra_cnt+inter_cnt)*100))
output_file.write('\t Intra-Sentence \t\t {} \t\t {:.4f}'.format(intra_cnt, intra_cnt/(intra_cnt + inter_cnt)*100))
output_file.write('\t Inter-Sentence \t\t {} \t\t {:.4f}'.format(inter_cnt, inter_cnt/(intra_cnt+inter_cnt)*100))
print("total: ", intra_cnt + inter_cnt)
print("test set")
output_file.write("test* set\n")
intra_cnt, inter_cnt = inter_intra_cnt('test.json')
print('\t Intra-Sentence \t\t {} \t\t {:.4f}'.format(intra_cnt, intra_cnt/(intra_cnt + inter_cnt)*100))
print('\t Inter-Sentence \t\t {} \t\t {:.4f}'.format(inter_cnt, inter_cnt/(intra_cnt+inter_cnt)*100))
output_file.write('\t Intra-Sentence \t\t {} \t\t {:.4f}'.format(intra_cnt, intra_cnt/(intra_cnt + inter_cnt)*100))
output_file.write('\t Inter-Sentence \t\t {} \t\t {:.4f}'.format(inter_cnt, inter_cnt/(intra_cnt+inter_cnt)*100))
print("total: ", intra_cnt + inter_cnt)