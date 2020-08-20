"""
##########################################################################
Usage:
Convert the prediction results to the DocRED format.

Steps:
1. mkdir ../data/DocRED & mkdir ../data/Dialogue
2. put the prediction file (e.g. test.preds) and rel2id.json under each directory
3. Run the script
DocRED:
python3 convert2docREDFormat.py --data DocRED

DialogRE:
python3 convert2docREDFormat.py --data Dialogue
##########################################################################
"""
import argparse
import json
import os

def convert2docred(pred_file, output_file, dataset):
    rel2id = json.load(open(os.path.join('../data', dataset, 'rel2id.json')))
    id2rel = {v: k for k, v in rel2id.items()}
    pred_file = os.path.join('../data', dataset, 'results', pred_file)
    output_file = os.path.join('../data', dataset, 'post_processed', output_file)
    output = []
    titleset = set()
    with open(pred_file) as f:
        cnt = 0
        for line in f:
            curline = line.split('|')
            index = cnt
            title = curline[0]
            if title not in titleset:
                titleset.add(title)
                cnt += 1
            h = int(curline[1])
            t = int(curline[2])
            r_idx = int(curline[-1].split(":")[1])
            r = id2rel[r_idx]
            output.append({'index': index, 'h_idx': h, 't_idx': t, 'r_idx': r_idx, 'r': r, 'title': title})
    f.close()
    if output:
        json.dump(output, open(output_file, "w"))

def main():
    """
    Main processing function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_prediction_file', type=str, default='test.preds')
    parser.add_argument('--output_file', type=str, default='test_index.json')
    parser.add_argument('--data', type=str, default='DocRED')
    args = parser.parse_args()
    convert2docred(args.input_prediction_file, args.output_file, args.data)

if __name__ == "__main__":
    main()