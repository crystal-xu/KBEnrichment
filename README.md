# Knowledge Base Enrichment in Conversational Domain
Implementations for my MSc dissertation. 

# Dataset

#### **DocRED**

Please download it [here](https://github.com/thunlp/DocRED/tree/master/data), provided by [DocRED: A Large-Scale Document-Level Relation Extraction Dataset](https://arxiv.org/abs/1906.06127). 

#### **DialogRE**

please download it [here](https://github.com/nlpdata/dialogre/tree/master/data), provided by [Dialogue-Based Relation Extraction](https://arxiv.org/abs/2004.08056). 

#### Pre-processing

DialogRE needs to be converted to the same format as DocRED.



# BiLSTM

Adapted from https://github.com/thunlp/DocRED/tree/master

Dataset and code for baselines for [DocRED: A Large-Scale Document-Level Relation Extraction Dataset](https://arxiv.org/abs/1906.06127v3)

Multiple entities in a document generally exhibit complex inter-sentence relations, and cannot be well handled by existing relation extraction (RE) methods that typically focus on extracting intra-sentence relations for single entity pairs. In order to accelerate the research on document-level RE, we introduce DocRED, a new dataset constructed from Wikipedia and Wikidata with three features: 

+ DocRED annotates both named entities and relations, and is the largest human-annotated dataset for document-level RE from plain text.
+ DocRED requires reading multiple sentences in a document to extract entities and infer their relations by synthesizing all information of the document.
+ Along with the human-annotated data, we also offer large-scale distantly supervised data, which enables DocRED to be adopted for both supervised and weakly supervised scenarios.



Code cloned from https://github.com/hongwang600/DocRed/tree/master/



# BERT-Embed

Adapted from https://github.com/hongwang600/DocRed/tree/master

# Sent-Model

Adapted from https://github.com/hongwang600/DocRed/tree/sent_level_enc

# Graph-EOG

Adapted from https://github.com/fenchri/edge-oriented-graph/tree/master

Adapted to the DocRED dataset

Source code for the paper "[Connecting the Dots: Document-level Relation Extraction with Edge-oriented Graphs](https://www.aclweb.org/anthology/D19-1498.pdf)" in EMNLP 2019.


## Environment

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

# Graph-LSR

Adapted from https://github.com/nanguoshun/LSR/tree/master

This repository is the PyTorch implementation of our LSR model with GloVe embeddings in ACL 2020 Paper 
"[Reasoning with Latent Structure Refinement for Document-Level Relation Extraction](https://arxiv.org/abs/2005.06312)".

## Requirement

```
python==3.6.7 
torch==1.3.1 + CUDA == 9.2 1.5.1
OR torch==1.5.1 + CUDA == 10.1
tqdm==4.29.1
numpy==1.15.4
spacy==2.1.3
networkx==2.4
```

## Data Proprocessing

After you download the dataset, please put the files train_annotated.json, dev.json and test.json to the ./data directory, and files in pre directory to the code/prepro_data. Run:

```
# cd code
# python3 gen_data.py 
```

## Training

In order to train the model, run:

```
# cd code
# python3 train.py
```

## Test

After the training process, we can test the model by:

```
python3 test.py
```

## Related Repo

Codes are adapted from the repo of the ACL2019 paper DocRED [DocRED: A Large-Scale Document-Level Relation Extraction Dataset](https://github.com/thunlp/DocRED).

## Citation

```
@inproceedings{nan2020lsr,
 author = {Guoshun, Nan and Zhijiang, Guo and  Ivan, Sekulić and Wei, Lu},
 booktitle = {Proc. of ACL},
 title = {Reasoning with Latent Structure Refinement for Document-Level Relation Extraction},
 year = {2020}
}
```



# BERT-LSR

It is the PyTorch implementation of the LSR model with BERT embeddings in ACL 2020 Paper 
"[Reasoning with Latent Structure Refinement for Document-Level Relation Extraction](https://arxiv.org/abs/2005.06312)".

This repository adapted the authors' implementation of LSR with GloVe embeddings: https://github.com/nanguoshun/LSR/tree/master

## Requirement

```
python==3.6.7 
torch==1.3.1 + CUDA == 9.2 1.5.1
OR torch==1.5.1 + CUDA == 10.1
tqdm==4.29.1
numpy==1.15.4
spacy==2.1.3
networkx==2.4
pytorch-transformers==1.2.0
```

## Dataset

**DocRED**:  please download it [here](https://github.com/thunlp/DocRED/tree/master/data), which are officially provided by [DocRED: A Large-Scale Document-Level Relation Extraction Dataset](https://arxiv.org/abs/1906.06127). 

**DialogRE**: please download it [here](https://github.com/nlpdata/dialogre/tree/master/data), which are officially provided by [Dialogue-Based Relation Extraction](https://arxiv.org/abs/2004.08056). 

## Data Proprocessing

**DocRED**: After you download the dataset, please put the files train_annotated.json, dev.json and test.json to the ./data directory, and files in pre directory to the code/prepro_data. Run:

```
# cd code
# python3 gen_data_bert.py 
```

**DialogRE**: After you download the dataset, please put the files train_annotated.json, dev.json and test.json to the ./data directory, and files in pre directory to the code/prepro_data. Run:

```
# cd code
# python3 gen_data_bert.py 
```

## Training

In order to train the model, run:

```
# cd code
# python3 train.py
```

## Test

After the training process, we can test the model by:

```
python3 test.py
```

## Citation

```
@inproceedings{nan2020lsr,
 author = {Guoshun, Nan and Zhijiang, Guo and  Ivan, Sekulić and Wei, Lu},
 booktitle = {Proc. of ACL},
 title = {Reasoning with Latent Structure Refinement for Document-Level Relation Extraction},
 year = {2020}
}
```



DialogRE
=====

Adapted from https://github.com/nlpdata/dialogre/tree/master

This repository maintains **DialogRE**, the first human-annotated dialogue-based relation extraction dataset (**Chinese** version coming soon).

* Paper: https://arxiv.org/abs/2004.08056

```
@inproceedings{yu2020dialogue,
  title={Dialogue-Based Relation Extraction},
  author={Yu, Dian and Sun, Kai and Cardie, Claire and Yu, Dong},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year={2020},
  url={https://arxiv.org/abs/2004.08056v1}
}
```

* ```kb/Fandom_triples```: relational triples from [Fandom](https://friends.fandom.com/wiki/Friends_Wiki).

* ```kb/matching_table.txt```: mapping from Fandom relational types to DialogRE relation types.

* ```bert``` folder: a re-implementation of BERT and BERT<sub>S</sub> baselines.

  1. Download and unzip BERT from [here](https://github.com/google-research/bert), and set up the environment variable for BERT by 
     ```export BERT_BASE_DIR=/PATH/TO/BERT/DIR```. 
  2. Copy the dataset folder ```data``` to ```bert/```.
  3. In ```bert```, execute ```python convert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path=$BERT_BASE_DIR/bert_model.ckpt --bert_config_file=$BERT_BASE_DIR/bert_config.json --pytorch_dump_path=$BERT_BASE_DIR/pytorch_model.bin```.
  4. To run and evaluate the BERT<sub>S</sub> baseline, execute the following commands in ```bert```:

  ```
  python run_classifier.py   --task_name berts  --do_train --do_eval   --data_dir .   --vocab_file $BERT_BASE_DIR/vocab.txt   --bert_config_file $BERT_BASE_DIR/bert_config.json   --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir berts_f1  --gradient_accumulation_steps 2
  rm berts_f1/model_best.pt && cp -r berts_f1 berts_f1c && python run_classifier.py   --task_name bertsf1c --do_eval   --data_dir .   --vocab_file $BERT_BASE_DIR/vocab.txt   --bert_config_file $BERT_BASE_DIR/bert_config.json   --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir berts_f1c  --gradient_accumulation_steps 2
  python evaluate.py --f1dev berts_f1/logits_dev.txt --f1test berts_f1/logits_test.txt --f1cdev berts_f1c/logits_dev.txt --f1ctest berts_f1c/logits_test.txt
  ```

  **Environment**:
  The code has been tested with Python 3.6 and PyTorch 1.0.
