#!/usr/bin/env python
'''
###############################################################
Usage: Get the distributions of entity-pair distances of part_of
instances on DocRE training, dev and test sets.

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
parser.add_argument('--output_dir', type = str, default='re_dist')

args = parser.parse_args()
input_dir = os.path.join(os.getcwd(), 'DocRED', args.input_dir)
output_dir = os.path.join(os.getcwd(), 'DocRED', args.output_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_filename = os.path.join(input_dir, 'train_annotated.json')
re_data = json.load(open(input_filename))

output_filename = os.path.join(output_dir, 'part_of_birth.txt')
output_json = os.path.join(output_dir, 'part_of_birth.json')

def find_pair_min_dist(ent1, ent2):
    dist = float('inf')
    for idx1, m1 in enumerate(ent1):
        for idx2, m2 in enumerate(ent2):
            cur_dist = abs(m1['sent_id'] - m2['sent_id'])
            dist = min(dist, cur_dist)
    return int(dist != 0)

cnt = 0
part_of_data = []
re_dist = {}
flag = False
cnt_1 = 0
for x in re_data:
    vertexSet = x['vertexSet']
    for label in x['labels']:
        h_idx = label['h']
        t_idx = label['t']
        if label['r'] == 'P361':
            evi_sents = label['evidence']
            cnt += 1
            part_of_data.append(x)
            dist = find_pair_min_dist(vertexSet[h_idx], vertexSet[t_idx])
            if dist not in re_dist:
                re_dist[dist] = 1
            else:
                re_dist[dist] += 1

sorted_re_dist = sorted(re_dist.items(), key=lambda x:x[1], reverse=True)

f = open(output_filename, 'w')
list = ['', 'Distance', 'Number', "Percentage"]
print('\t'.join(list))
f.write('\t'.join(list) + '\n')
distList = []
numberList = []
for item in sorted_re_dist:
    #print("\t{}   \t   {}   \t   {:.4f}".format(item[0], item[1], item[1] / cnt * 100))
    #f.write("\t{}   \t   {}\t   {:.4f}\n".format(item[0], item[1], item[1] / cnt * 100))
    distList.append(item[0])
    numberList.append(item[1])

print('-----------------------------------------------------')
f.write('------------------------------------------------------')

def get_dist_num(start, end):
    cnt_dist = 0
    for item in sorted_re_dist[start:end]:
        cnt_dist += item[1]
    return cnt_dist
# 0
cnt_dist = get_dist_num(0, 1)
print("\t0   \t   {}   \t   {:.4f}".format(cnt_dist, cnt_dist / cnt * 100))
f.write("\t0   \t   {}\t   {:.4f}\n".format(cnt_dist, cnt_dist / cnt * 100))

# 1-3
cnt_dist = get_dist_num(1, 4)
print("\t1-3   \t   {}   \t   {:.4f}".format(cnt_dist, cnt_dist / cnt * 100))
f.write("\t1-3   \t   {}\t   {:.4f}\n".format(cnt_dist, cnt_dist / cnt * 100))

# >3
cnt_dist = get_dist_num(3, len(sorted_re_dist))
print("\t>3   \t   {}   \t   {:.4f}".format(cnt_dist, cnt_dist / cnt * 100))
f.write("\t>3   \t   {}\t   {:.4f}\n".format(cnt_dist, cnt_dist / cnt * 100))

f.close()

json.dump(part_of_data, open(output_json, 'w'))
print("total inst: ", cnt)


