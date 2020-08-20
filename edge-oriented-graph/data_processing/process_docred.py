#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: fenia
"""

import os
import re
from tqdm import tqdm
from recordtype import recordtype
from collections import OrderedDict
import argparse
from tools import adjust_offsets, find_mentions, find_cross, fix_sent_break, convert2sent, generate_pairs
from readers import *
import json


TextStruct = recordtype('TextStruct', 'pmid txt')
EntStruct = recordtype('EntStruct', 'pmid name off1 off2 type kb_id sent_no word_id bio')
RelStruct = recordtype('RelStruct', 'pmid type arg1 arg2')

def processDocRED(input_file, output_file):
    # Process
    positive, negative = 0, 0
    origin_data = json.load(open(input_file))
    abstracts = OrderedDict()
    entities = OrderedDict()
    relations = OrderedDict()
    unique_entities = OrderedDict()

    type1 = ['PER', 'ORG', 'LOC', 'TIME', 'NUM', 'MISC']
    type2 = ['PER', 'ORG', 'LOC', 'TIME', 'NUM', 'MISC']
    # title = x['title']
    # sents = x['sents']
    # vertexSet = x['vertexSet']
    # labels = x['labels']
    unique_entities_dict = OrderedDict()
    for docIdx, data in enumerate(origin_data):
        #docIdx = str(docIdx)
        docIdx = data['title']
        abstracts[docIdx] = []
        for sent in data['sents']:
            abstracts[docIdx].append(" ".join(sent))
        unique_entities_dict[docIdx] = OrderedDict()
        for entIdx, ent in enumerate(data['vertexSet']):
            for men in ent:
                entIdx = str(entIdx)
                if '\n' in men['name']:
                    men['name'] = men['name'].split('\n')[1]
                #tmp = [EntStruct(docIdx, men['name'], -1, -1, men['type'], entIdx, men['sent_id'], men['pos'], [])]
                if docIdx not in entities:
                    entities[docIdx] = [EntStruct(docIdx, men['name'], -1, -1, men['type'], [entIdx], men['sent_id'], men['pos'], [])]
                else:
                    entities[docIdx] += [EntStruct(docIdx, men['name'], -1, -1, men['type'], [entIdx], men['sent_id'], men['pos'], [])]
                if tuple([entIdx]) not in unique_entities_dict[docIdx]:
                    unique_entities_dict[docIdx][tuple([entIdx])] = [EntStruct(docIdx, men['name'], -1, -1, men['type'], [entIdx], men['sent_id'], men['pos'], [])]
                else:
                    unique_entities_dict[docIdx][tuple([entIdx])] += [EntStruct(docIdx, men['name'], -1, -1, men['type'], [entIdx], men['sent_id'], men['pos'], [])]
        if 'labels' not in data:
            continue
        for rel in data['labels']:
            if docIdx not in relations:
                relations[docIdx] = [RelStruct(docIdx, str(rel['r']), tuple([str(rel['h'])]), tuple([str(rel['t'])]))]
            else:
                relations[docIdx] += [RelStruct(docIdx, str(rel['r']), tuple([str(rel['h'])]), tuple([str(rel['t'])]))]

    with open(output_file + '.data', 'w') as data_out:
        pbar = tqdm(list(abstracts.keys()))
        for i in pbar:
            pbar.set_description("Processing Doc_ID {}".format(i))
        ''' Generate Pairs '''
        #for i in range(len(origin_data)):
        for data in origin_data:
            i = data['title']
            unique_entities = unique_entities_dict[i]
            if i in relations:
                pairs = generate_pairs(unique_entities, type1, type2, relations[i])
            else:
                pairs = generate_pairs(unique_entities, type1, type2, [])   # generate only negative pairs
            # print('{}\t{}'.format(i, '|'.join(abstracts[i])))
            # 'pmid type arg1 arg2 dir cross'
            data_out.write('{}\t{}'.format(i, '|'.join(abstracts[i])))

            for args_, p in pairs.items():
                if p.type != '1:NR:2':
                    positive += 1
                elif p.type == '1:NR:2':
                    negative += 1

                data_out.write('\t{}\t{}\t{}\t{}-{}\t{}-{}'.format(p.type, p.dir, p.cross, p.closest[0].word_id[0],
                                                                                           p.closest[0].word_id[-1]+1,
                                                                                           p.closest[1].word_id[0],
                                                                                           p.closest[1].word_id[-1]+1))
                data_out.write('\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                    '|'.join([g for g in p.arg1]),
                    '|'.join([e.name for e in unique_entities[p.arg1]]),
                    unique_entities[p.arg1][0].type,
                    ':'.join([str(e.word_id[0]) for e in unique_entities[p.arg1]]),
                    ':'.join([str(e.word_id[-1] + 1) for e in unique_entities[p.arg1]]),
                    ':'.join([str(e.sent_no) for e in unique_entities[p.arg1]])))

                data_out.write('\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                    '|'.join([g for g in p.arg2]),
                    '|'.join([e.name for e in unique_entities[p.arg2]]),
                    unique_entities[p.arg2][0].type,
                    ':'.join([str(e.word_id[0]) for e in unique_entities[p.arg2]]),
                    ':'.join([str(e.word_id[-1] + 1) for e in unique_entities[p.arg2]]),
                    ':'.join([str(e.sent_no) for e in unique_entities[p.arg2]])))
            data_out.write('\n')
    print('Total positive pairs:', positive)
    print('Total negative pairs:', negative)
    return 0

def main():
    """
    Main processing function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', '-i', type=str)
    parser.add_argument('--output_file', '-o', type=str)
    parser.add_argument('--data', '-d', type=str)
    args = parser.parse_args()

    if args.data == 'DocRED':
        processDocRED(args.input_file, args.output_file)

    else:
        print('Dataset non-existent.')
        sys.exit()


if __name__ == "__main__":
    main()
