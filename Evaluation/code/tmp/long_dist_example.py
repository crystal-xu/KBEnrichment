#!/usr/bin/env python
import os.path
import json
import argparse

parser = argparse.ArgumentParser()
#cwd = os.getcwd()
parser.add_argument('--input_dir', type = str, default='re_data')
parser.add_argument('--output_dir', type = str, default='re_dist')

args = parser.parse_args()
input_dir = os.path.join(os.getcwd(), args.input_dir)
output_dir = os.path.join(os.getcwd(), args.output_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

input_filename = os.path.join(input_dir, 'train_annotated.json')
re_data = json.load(open(input_filename))

output_filename = os.path.join(output_dir, 'long_dist_example.txt')
output_json = os.path.join(output_dir, 'long_dist_example.json')

def maxDist(sents):
    sents = sorted(sents)
    dist = sents[-1] - sents[0]
    return dist

cnt = 0
ignore_cnt = 0
long_dist_data = []
flag = False
for x in re_data:
    cnt += len(x['labels'])
    for label in x['labels']:
        evi_sents = label['evidence']
        if len(evi_sents) > 1:
            dist = maxDist(evi_sents)
            if dist == 20:
                long_dist_data.append(x)
                flag = True
                break
    if flag:
        break

with open(output_filename, 'w') as f:
    for data in long_dist_data:
        title = data['title']
        f.write(title + '\n')
        for idx, sent in enumerate(data['sents']):
            f.write('[{}]\t{}\n'.format(str(idx), " ".join(sent)))
        for idx, ent in enumerate(data['vertexSet']):
            mens = []
            mstarts = []
            mends = []
            sent_ids = []
            type = None
            for men in ent:
                mens.append(men['name'])
                mstarts.append(str(men['pos'][0]))
                mends.append(str(men['pos'][-1]))
                type = men['type']
                sent_ids.append(str(men['sent_id']))
            f.write('[{}]\t{}\t{}\t{}\t{}\t{}\n'.format(str(idx), type, "|".join(mens), ":".join(mstarts), ":".join(mends), ":".join(sent_ids)))
        for label in data['labels']:
            evi_sents = label['evidence']
            evi_sents = list(map(lambda x: str(x), evi_sents))
            f.write('{}\t{}\t{}\t{}\n'.format(str(label['h']), str(label['t']), label['r'], ":".join(evi_sents)))
f.close()

json.dump(long_dist_data, open(output_json, 'w'))


