#!/usr/bin/env python
'''
###############################################################
Usage: Get the distributions of relation distances
for two relation types: date_of_birth and part_of.

Run the script:
python3 get_date_of_birth_distri.py --inputfile train_annotated.json|dev.json|test.json
--type date_of_birth|part_of
###############################################################
'''
import os.path
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type = str, default='re_data')
parser.add_argument('--output_dir', type = str, default='re_dist')
parser.add_argument('--inputfile', type = str, default='train_annotated.json')
parser.add_argument('--type', type=str, default='date_of_birth')
#parser.add_argument('--type', type=str, default='part_of')

args = parser.parse_args()
input_dir = os.path.join(os.getcwd(), 'DocRED', args.input_dir)
output_dir = os.path.join(os.getcwd(), 'DocRED', args.output_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_filename = os.path.join(input_dir, args.inputfile)
re_data = json.load(open(input_filename))
output_filename = os.path.join(output_dir, 'date_of_birth.txt')
output_json = os.path.join(output_dir, 'data_of_birth.json')
if args.type == 'date_of_birth':
    re_type = 'P569'
elif args.type == 'part_of':
    re_type = 'P361'

def find_pair_min_dist(ent1, ent2):
    dist = float('inf')
    for idx1, m1 in enumerate(ent1):
        for idx2, m2 in enumerate(ent2):
            cur_dist = abs(m1['sent_id'] - m2['sent_id'])
            dist = min(dist, cur_dist)
    return int(dist != 0)

cnt = 0
data = []
re_dist = {}
flag = False
for x in re_data:
    vertexSet = x['vertexSet']
    for label in x['labels']:
        h_idx = label['h']
        t_idx = label['t']
        if label['r'] == re_type:
            evi_sents = label['evidence']
            cnt += 1
            data.append(x)
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


print('-----------------------------------------------------')
f.write('------------------------------------------------------')
# Intra-
cnt_dist = get_dist_num(0, 1)
print('\t Intra-Sentence \t\t {} \t\t {:.4f}'.format(cnt_dist, cnt_dist / cnt * 100))
f.write('\t Intra-Sentence \t\t {} \t\t {:.4f}'.format(cnt_dist, cnt_dist / cnt * 100))

# Inter-
cnt_dist = get_dist_num(1, len(sorted_re_dist))
print('\t Inter-Sentence \t\t {} \t\t {:.4f}'.format(cnt_dist, cnt_dist / cnt * 100))
f.write('\t Inter-Sentence \t\t {} \t\t {:.4f}'.format(cnt_dist, cnt_dist / cnt * 100))

f.close()

json.dump(data, open(output_json, 'w'))
print("total inst: ", cnt)


