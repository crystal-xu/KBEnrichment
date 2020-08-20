#!/usr/bin/env python
'''
###############################################################
Usage: Get the distributions of different relation distances.

Run the script:
python3 get_re_dist_distri.py --data DocRED/Dialogue
###############################################################
'''
import os.path
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type = str, default='re_data')
parser.add_argument('--output_dir', type = str, default='re_dist')
parser.add_argument('--data', type=str, default='Dialogue')

args = parser.parse_args()
input_dir = os.path.join(os.getcwd(), args.data, args.input_dir)
output_dir = os.path.join(os.getcwd(), args.data, args.output_dir)

re_dist_dict = {}

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_filename = os.path.join(input_dir, 'dev.json')
re_data = json.load(open(input_filename))

output_filename = os.path.join(output_dir, 're_dist_distri.txt')
output_file = open(output_filename, 'w')

def find_pair_min_dist(ent1, ent2):
    dist = float('inf')
    for idx1, m1 in enumerate(ent1):
        for idx2, m2 in enumerate(ent2):
            cur_dist = abs(m1['sent_id'] - m2['sent_id'])
            dist = min(dist, cur_dist)
    return dist

cnt = 0
ignore_cnt = 0
for x in re_data:
    vertexSet = x['vertexSet']
    cnt += len(x['labels'])
    for label in x['labels']:
        r = label['r']
        h_idx = label['h']
        t_idx = label['t']
        dist = find_pair_min_dist(vertexSet[h_idx], vertexSet[t_idx])
        if dist not in re_dist_dict:
            re_dist_dict[dist] = 1
        else:
            re_dist_dict[dist] += 1
print("Total: {}".format(cnt))
output_file.write("Total: {}\n".format(cnt))
print("Intra-sentential relations: {}".format(re_dist_dict[0]))
output_file.write("Intra-sentential relations: {}\n".format(re_dist_dict[0]))
inter_cnt = cnt - re_dist_dict[0]
print("Inter-sentential relations: {}".format(cnt - re_dist_dict[0]))
output_file.write("Inter-sentential relations: {}\n".format(cnt - re_dist_dict[0]))

re_dist_ratio_dict = {k: v for k, v in re_dist_dict.items()}
sorted_re_dist_ratio = sorted(re_dist_ratio_dict.items())

list = ['', 'Distance', 'Number', 'Percentage']
print('\t'.join(list))
output_file.write('\t'.join(list) + '\n')

distList = []
numberList = []
for item in sorted_re_dist_ratio[0:]:
    print("\t{}   \t   {}   \t   {:.4f}".format(item[0], item[1], item[1]/cnt*100))
    output_file.write("\t{}   \t   {}\t   {:.4f}\n".format(item[0], item[1], item[1]/cnt*100))
    distList.append(item[0])
    numberList.append(item[1])

print('-----------------------------------------------------')
output_file.write('------------------------------------------------------')
def get_dist_num(start, end):
    cnt_dist = 0
    for item in sorted_re_dist_ratio[start:end]:
        cnt_dist += item[1]
    return cnt_dist
# 0
cnt_dist = get_dist_num(0, 1)
print("\t0   \t   {}   \t   {:.4f}".format(cnt_dist, cnt_dist / cnt * 100))
output_file.write("\t0   \t   {}\t   {:.4f}\n".format(cnt_dist, cnt_dist / cnt * 100))

cnt_dist = get_dist_num(1, 2)
print("\t1   \t   {}   \t   {:.4f}".format(cnt_dist, cnt_dist / cnt * 100))
output_file.write("\t1   \t   {}\t   {:.4f}\n".format(cnt_dist, cnt_dist / cnt * 100))

cnt_dist = get_dist_num(2, 6)
print("\t2-5   \t   {}   \t   {:.4f}".format(cnt_dist, cnt_dist / cnt * 100))
output_file.write("\t2-5   \t   {}\t   {:.4f}\n".format(cnt_dist, cnt_dist / cnt * 100))

cnt_dist = get_dist_num(6, len(sorted_re_dist_ratio))
print("\t>5   \t   {}   \t   {:.4f}".format(cnt_dist, cnt_dist / cnt * 100))
output_file.write("\t>5   \t   {}\t   {:.4f}\n".format(cnt_dist, cnt_dist / cnt * 100))


output_file.close()
print("total re num: ", cnt)

