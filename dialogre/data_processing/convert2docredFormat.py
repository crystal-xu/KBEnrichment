##!/usr/bin/env python
'''
##############################################################################
Created by Yuwei Xu.
Usage: Convert DialogRE to DocRED data format.

Run the script:
python3 convert2docredFormat.py --inputfile train.json --outputfile dev_train.json

##############################################################################
'''
import spacy
from spacy.tokenizer import Tokenizer
import nltk
import argparse
import os
import json
from collections import OrderedDict
from spacy.symbols import ORTH, LEMMA, POS

nltk.download('punkt')
from nltk.stem.porter import *

import inflect
p = inflect.engine()
stemmer = PorterStemmer()

custom_nlp = spacy.load('en_core_web_sm')
infix_re = re.compile(r'[^a-z0-9]')

def customize_tokenizer(nlp):
    return Tokenizer(nlp.vocab,
                     infix_finditer=infix_re.finditer
                    )

def match_ent_in_sent(ent, sent, sent_id):
    match_ids = []
    for idx in range(len(sent) - len(ent) + 1):
        flag = True
        for i in range(len(ent)):
            if stemmer.stem(sent[idx + i].lower()) == stemmer.stem(ent[i].lower()):
                continue
            else:
                flag = False
                break
        if flag:
           match_ids.append(tuple((idx, idx + len(ent), sent_id)))
    return match_ids


def convert2docredFormat(input_file, output_file):
    cnt_no_rel = 0
    if 'train' in input_file:
        dataset = 'train'
    elif 'dev' in input_file:
        dataset = 'dev'
    elif 'test' in input_file:
        dataset = 'test'
    input_json = os.path.join('../data', input_file)
    output_json = os.path.join('../data/processed', output_file)
    dialogues = json.load(open(input_json))
    docred_data = []
    custom_nlp.tokenizer = customize_tokenizer(custom_nlp)
    # Add some special cases during tokenization
    custom_nlp.tokenizer.add_special_case(u'Dr.',
                                   [
                                       {
                                           ORTH: u'Dr.',
                                           LEMMA: u'Dr.',
                                           POS: u'NOUN'}
                                   ])
    custom_nlp.tokenizer.add_special_case(u"Mrs.",
                                   [
                                       {
                                           ORTH: u"Mrs.",
                                           LEMMA: u"Mrs.",
                                           POS: u'NOUN'}
                                   ])
    custom_nlp.tokenizer.add_special_case(u"Mr.",
                                   [
                                       {
                                           ORTH: u"Mr.",
                                           LEMMA: u"Mr.",
                                           POS: u'NOUN'}
                                   ])
    custom_nlp.tokenizer.add_special_case(u"Café",
                                   [
                                       {
                                           ORTH: u"Café",
                                           LEMMA: u"Café",
                                           POS: u'NOUN'}
                                   ])

    custom_nlp.tokenizer.add_special_case(u"TV",
                                   [
                                       {
                                           ORTH: u"TV",
                                           LEMMA: u"TV",
                                           POS: u'TV'}
                                   ])

    doc_len = len(dialogues)
    for idx in range(0, doc_len):
        data = dialogues[idx]
        mentionDict = OrderedDict()
        relDict = OrderedDict()
        token2mention = OrderedDict()
        mention2token = OrderedDict()
        mention2type = OrderedDict()
        labels = []
        sents = []
        vertexes = []
        docred_tmp = OrderedDict()
        docred_tmp['title'] = dataset + ' ' + str(idx)
        # sents
        for sent in data[0]:
            tmp = []
            sent = re.sub(r"(\.+)", '. ', sent)
            sent = re.sub(r"…", '… ', sent)
            sent = re.sub(r"\"", "\" ", sent)
            sent = re.sub(r"\'", "\' ", sent)
            sent = re.sub(r"\?+", "? ", sent)
            sent = re.sub(r"!+", "! ", sent)
            sent = re.sub(r"-", "- ", sent)
            newSent = []
            for word in sent.split(" "):
                if word[:-1].isalpha() and len(word) >= 3:
                    newSent.append(word[0] + word[1:].lower())
                else:
                    newSent.append(word)

            sent = " ".join(newSent)
            for token in custom_nlp(sent):
                if token.text != ' ':
                    tmp.append(token.text)
            sents.append(tmp)
        for item in data[1]:

            ent2 = item['y']
            ent2_type = item['y_type']
            ent1 = item['x']
            ent1_type = item['x_type']
            rid = item['rid']
            r = item['r']
            if ent1 == "big spender" or ent2 == "big spender":
                continue
            mention2type[ent1] = ent1_type
            mention2type[ent2] = ent2_type
            if (ent1, ent2) not in relDict:
                relDict[(ent1, ent2)] = r
        for ent in mention2type:
            mention = ent
            ent = re.sub(r"-", "- ", ent)
            newEnt = []
            for word in ent.split(" "):
                if word[:-1].isalpha() and len(word) >= 3:
                    newEnt.append(word[0] + word[1:].lower())
                else:
                    newEnt.append(word)
            ent = " ".join(newEnt)

            for sent_id, sent in enumerate(sents):
                tmp = []
                for item in custom_nlp(ent):
                    tmp.append(item.text)
                ent = " ".join(tmp)

                #  handle specific cases manually
                if mention == "Frank Jr.Jr.":
                    tmp = ['Frank', 'Jr', '.', 'Jr', '.']
                    ent = ' '.join(tmp)
                if mention == "G.Stephanopoulos":
                    tmp = ['G', '.', 'Stephanopoulos']
                    ent = ' '.join(tmp)
                if mention == "Mike \"Gandolf\" Ganderson":
                    tmp = ['Mike', '"', 'Gandolf', '"', 'Ganderson']
                    ent = ' '.join(tmp)
                if mention == "Howard-the-\"I-win\"-guy":
                    tmp = ['Howard', '-', 'the', '-', '"', 'I', '-', 'win', '"', '-', 'guy']
                    ent = ' '.join(tmp)

                match_ids = match_ent_in_sent(tmp, sent, sent_id)
                if not len(match_ids):
                    continue
                token2mention[ent] = mention
                mention2token[mention] = ent
                if ent not in mentionDict:
                    mentionDict[ent] = []
                mentionDict[ent].append(match_ids)
        print("idx: ", idx)
        # vertexSet
        cnt = 0
        ent2id = OrderedDict()
        for name, ids in mentionDict.items():
            tmpList = []
            for elem in ids:
                if not len(elem):
                    print(name)
                    continue
                tmpDict = OrderedDict()
                for x in elem:
                    tmpDict['name'] = name
                    tmpDict['pos'] = list((x[0], x[1]))
                    tmpDict['sent_id'] = x[-1]
                    tmpDict['type'] = mention2type[token2mention[name]]
                    tmpList.append(tmpDict)
            vertexes.append(tmpList)
            ent2id[token2mention[name]] = cnt
            cnt += 1
        docred_tmp['vertexSet'] = vertexes
        for pairs, rel in relDict.items():
            for item in rel:
                tmpDict = OrderedDict()
                tmpDict['r'] = item
                tmpDict['h'] = ent2id[pairs[0]]
                tmpDict['t'] = ent2id[pairs[1]]
                tmpDict['evidence'] = []
                labels.append(tmpDict)
                if item == 'unanswerable':
                    cnt_no_rel += 1
        docred_tmp['labels'] = labels
        docred_tmp['sents'] = sents
        docred_data.append(docred_tmp)
    json.dump(docred_data, open(output_json, 'w'))
    print("unanswerable: {}".format(cnt_no_rel))

    return 0

def main():
    """
    Main processing function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='test.json')
    parser.add_argument('--output_file', type=str, default='docred_test.json')
    args = parser.parse_args()
    convert2docredFormat(args.input_file, args.output_file)

if __name__ == "__main__":
    main()