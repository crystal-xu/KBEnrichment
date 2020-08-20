# Edge-oriented Graph
Cloned from https://github.com/fenchri/edge-oriented-graph/tree/master

Adapted to the DocRED dataset

Source code for the paper "[Connecting the Dots: Document-level Relation Extraction with Edge-oriented Graphs](https://www.aclweb.org/anthology/D19-1498.pdf)" in EMNLP 2019.


### Environment
`$ pip3 install -r requirements.txt`  

## Datasets & Pre-processing
Download the datasets
```
$ mkdir data && cd data
$ mkdir DocRED && mkdir Dialogue
$ # put dev_train.json dev_dev.json dev_test.json of the two datasets in each directory
$ cd ..
```

In order to process the datasets, they should first be transformed into the PubTator format. The run the processing scripts as follows:
```
$ sh process_docred.sh #DocRED
$ sh process_dialogue.sh #DialogRE
```

In order to get the data statistics run:
```
# DocRED
python3 statistics.py --data ../data/DocRED/processed/dev_train.data
python3 statistics.py --data ../data/DocRED/processed/dev_dev.data
python3 statistics.py --data ../data/DocRED/processed/dev_test.data

# DialogRE
python3 statistics.py --data ../data/Dialogue/processed/dev_train.data
python3 statistics.py --data ../data/Dialogue/processed/dev_dev.data
python3 statistics.py --data ../data/Dialogue/processed/dev_test.data
```
This will additionally generate the gold-annotation file in the same folder with suffix `.gold`.


## Usage
Run the main script from training and testing as follows. Select gpu -1 for cpu mode.  

**DocRED/DialogRE**: Train the model on the training set and evaluate on the dev set, in order to identify the best training epoch.
For testing, evaluate on the test set.

In order to ensure the usage of early stopping criterion, use the `--early_stop` option.
If during training early stopping is not triggered, the maximum epoch (specified in the config file) will be used.

Otherwise, if you want to train up to a specific epoch, use the `--epoch epochNumber` option without early stopping.
The maximum stopping epochs can be defined by the `--epoch` option.

For example, in the DocRED dataset:
```
$ cd src/
$ python3 eog.py --config ../configs/parameters_docred.yaml --train --gpu 0 --early_stop       # using early stopping
$ python3 eog.py --config ../configs/parameters_docred.yaml --train --gpu 0 --epoch 15         # train until the 15th epoch *without* early stopping
$ python3 eog.py --config ../configs/parameters_docred.yaml --train --gpu 0 --epoch 15 --early_stop  # set both early stop and max epoch

$ python3 eog.py --config ../configs/parameters_docred.yaml --test --gpu 0
```

All necessary parameters can be stored in the yaml files inside the configs directory.
The following parameters can be also directly given as follows:
```
usage: eog.py [-h] --config CONFIG [--train] [--test] [--gpu GPU]
              [--walks WALKS] [--window WINDOW] [--edges [EDGES [EDGES ...]]]
              [--types TYPES] [--context CONTEXT] [--dist DIST] [--example]
              [--seed SEED] [--early_stop] [--epoch EPOCH]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Yaml parameter file
  --train               Training mode - model is saved
  --test                Testing mode - needs a model to load
  --gpu GPU             GPU number
  --walks WALKS         Number of walk iterations
  --window WINDOW       Window for training (empty processes the whole
                        document, 1 processes 1 sentence at a time, etc)
  --edges [EDGES [EDGES ...]]
                        Edge types
  --types TYPES         Include node types (Boolean)
  --context CONTEXT     Include MM context (Boolean)
  --dist DIST           Include distance (Boolean)
  --example             Show example
  --seed SEED           Fixed random seed number
  --early_stop          Use early stopping
  --epoch EPOCH         Maximum training epoch
```

### Post-processing
In order to evaluate the results, the prediction file "test.preds" need to be converted to the same format as DocRED:
```
$ # DocRED
$ mkdir ../data/DocRED 
$ # put the test.preds and rel2id.json under the directory
$ python3 convert2DocREDFormat --data DocRED
$ # DialogRE
$ mkdir ../data/Dialogue 
$ # put the test.preds and rel2id.json under the directory
$ python3 convert2DocREDFormat --data Dialogue
```


### Acknowledgement

The initial idea and code own to the following paper:
```
@inproceedings{christopoulou2019connecting,  
title = "Connecting the Dots: Document-level Neural Relation Extraction with Edge-oriented Graphs",  
author = "Christopoulou, Fenia and Miwa, Makoto and Ananiadou, Sophia",  
booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",  
year = "2019",  
publisher = "Association for Computational Linguistics",  
pages = "4927--4938"  
}  
```
