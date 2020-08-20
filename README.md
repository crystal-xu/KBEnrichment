# Knowledge Base Enrichment in Conversational Domain
This repository is the implementations for my MSc dissertation. 

# Dataset

#### **DocRED**

Please download it [here](https://github.com/thunlp/DocRED/tree/master/data), provided by [DocRED: A Large-Scale Document-Level Relation Extraction Dataset](https://arxiv.org/abs/1906.06127). 

#### **DialogRE**

Please download it [here](https://github.com/nlpdata/dialogre/tree/master/data), provided by [Dialogue-Based Relation Extraction](https://arxiv.org/abs/2004.08056). 

#### Pre-processing

DialogRE needs to be converted to the same format as DocRED.

- Enter the directory:

   [```cd dialogre/data_processing```](https://github.com/crystal-xu/KBEnrichment/tree/master/dialogre/data_processing)

- Run the shell script:

   ```source process_docred.sh```

  Three documents will be generated under the directory [```../data/processed```](https://github.com/crystal-xu/KBEnrichment/tree/master/dialogre/data/processed):

   ```train_annotated.json```, ```dev.json```, ```test.json```

  Note: their names are the same as DocRED for convenience.

- We also put the three pre-processed documents under the directory [```../data/processed```](https://github.com/crystal-xu/KBEnrichment/tree/master/dialogre/data/processed).



# BiLSTM

**Main directory**:

 [```cd docred```](https://github.com/crystal-xu/KBEnrichment/tree/master/DocRED)

**Adapted from**:

 https://github.com/thunlp/DocRED/tree/master

**Reference paper:** 

 [DocRED: A Large-Scale Document-Level Relation Extraction Dataset](https://arxiv.org/abs/1906.06127v3)

## Requirements and Installation

python3

pytorch>=1.0

```
pip3 install -r requirements.txt
```

## preprocessing data

**DocRED**

Download metadata from [TsinghuaCloud](https://cloud.tsinghua.edu.cn/d/99e1c0805eb64736af95/) or [GoogleDrive](https://drive.google.com/drive/folders/1Ri3LIILKKBi3aBJjUVCOBpGX5PpONHRK) for baseline method and put them into [```prepro_data```](https://github.com/crystal-xu/KBEnrichment/tree/master/DocRED/code/prepro_data) folder.

**DialogRE**

Replace the  ```rel2id.json``` under [```prepro_data```](https://github.com/crystal-xu/KBEnrichment/tree/master/DocRED/code/prepro_data)  with  [```dialogre/data_processing/rel2id.json```](https://github.com/crystal-xu/KBEnrichment/blob/master/dialogre/data/rel2id.json)

- Run the script:

```
$ cd code
$ python3 gen_data.py --in_path ../data --out_path prepro_data
```

## Train

```
$ cd code
$ CUDA_VISIBLE_DEVICES=0 python3 train.py --model_name BiLSTM --save_name checkpoint_BiLSTM --train_prefix dev_train --test_prefix dev_dev
```

**Note: **change the [self.relation_num](https://github.com/crystal-xu/KBEnrichment/blob/db3a845c3c4d756f58ad2e9d2a223586d3096302/DocRED/code/config/Config.py#L54) to 37 for **DialogRE** 

## Test

```
$ cd code
$ CUDA_VISIBLE_DEVICES=0 python3 test.py --model_name BiLSTM --save_name checkpoint_BiLSTM --train_prefix dev_train --test_prefix dev_dev --input_theta 0.3601
```



# BERT-Embed

**Main directory**:

 [```cd DocRed-BERT```](https://github.com/crystal-xu/KBEnrichment/tree/master/DocRed-BERT)

**Adapted from**:

 https://github.com/hongwang600/DocRed/tree/master

**Reference paper:**

  [Fine-tune Bert for DocRED with Two-step Process](https://arxiv.org/abs/1909.11898)

**Note**: Please refer to [BiLSTM](https://github.com/crystal-xu/KBEnrichment/tree/master/DocRED) for preprocessing data, train, and test.



# Sent-Model

**Main directory**:

 [```cd DocRed-BERT```](https://github.com/crystal-xu/KBEnrichment/tree/master/DocRed-BERT)

**Adapted from**:

 https://github.com/hongwang600/DocRed/tree/sent_level_enc

**Reference paper:**  

[Fine-tune Bert for DocRED with Two-step Process](https://arxiv.org/abs/1909.11898)

**Note**: Please refer to [BiLSTM](https://github.com/crystal-xu/KBEnrichment/tree/master/DocRED) for preprocessing data, train, and test.



# Graph-LSR

**Main directory**: [```cd LSR```](https://github.com/crystal-xu/KBEnrichment/tree/master/LSR)

**Adapted from** https://github.com/nanguoshun/LSR/tree/master

**Reference paper**:
[Reasoning with Latent Structure Refinement for Document-Level Relation Extraction](https://arxiv.org/abs/2005.06312)

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

**DocRED**

Download metadata from [TsinghuaCloud](https://cloud.tsinghua.edu.cn/d/99e1c0805eb64736af95/) or [GoogleDrive](https://drive.google.com/drive/folders/1Ri3LIILKKBi3aBJjUVCOBpGX5PpONHRK) for baseline method and put them into ```prepro_data``` folder.

**DialogRE**

Replace the  ```rel2id.json``` under ```prepro_data``` with  [```dialogre/data_processing/rel2id.json```](https://github.com/crystal-xu/KBEnrichment/blob/master/dialogre/data/rel2id.json)

- Run the script:

```
$ cd code
$ python3 gen_data.py 
```

## Training

In order to train the model, run:

```
$ cd code
$ python3 train.py
```

**Note: **change the [self.relation_num](https://github.com/crystal-xu/KBEnrichment/blob/db3a845c3c4d756f58ad2e9d2a223586d3096302/LSR/code/config/Config.py#L77) to 37 for **DialogRE** 

## Test

After the training process, we can test the model by:

```
python3 test.py
```



# BERT-LSR

**Main directory**: [```cd LSR_BERT```](https://github.com/crystal-xu/KBEnrichment/tree/master/LSR_BERT)

**Adapted from** https://github.com/nanguoshun/LSR/tree/master

**Reference paper**:
[Reasoning with Latent Structure Refinement for Document-Level Relation Extraction](https://arxiv.org/abs/2005.06312)

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

## 

## Data Proprocessing

**DocRED**

Download metadata from [TsinghuaCloud](https://cloud.tsinghua.edu.cn/d/99e1c0805eb64736af95/) or [GoogleDrive](https://drive.google.com/drive/folders/1Ri3LIILKKBi3aBJjUVCOBpGX5PpONHRK) for baseline method and put them into ```prepro_data``` folder.

- Run the script

```
$ cd code
$ python3 gen_data.py 
```

**DialogRE**

Replace the  ```rel2id.json``` under ```prepro_data``` with  [```dialogre/data_processing/rel2id.json```](https://github.com/crystal-xu/KBEnrichment/blob/master/dialogre/data/rel2id.json)

- Run the script

```
$ cd code
$ python3 gen_data_bert.py 
```

## Training

In order to train the model, run:

```
$ cd code
$ python3 train.py
```

## Test

After the training process, we can test the model by:

```
python3 test.py
```



# Graph-EOG

**Main directory**:

[```cd edge-oriented-graph```](https://github.com/crystal-xu/KBEnrichment/tree/master/edge-oriented-graph)

**Adapted from**:

 https://github.com/fenchri/edge-oriented-graph/tree/master

**Reference paper**:

 [Connecting the Dots: Document-level Relation Extraction with Edge-oriented Graphs](https://www.aclweb.org/anthology/D19-1498/)


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

In order to process the datasets, two datasets should first be transformed into the PubTator format. 

Run the processing scripts as follows:

```
$ sh process_docred.sh #DocRED
$ sh process_dialogue.sh #DialogRE
```

In order to get the data statistics run:

- DocRED

```
python3 statistics.py --data ../data/DocRED/processed/dev_train.data
python3 statistics.py --data ../data/DocRED/processed/dev_dev.data
python3 statistics.py --data ../data/DocRED/processed/dev_test.data
```

- DialogRE

```
python3 statistics.py --data ../data/Dialogue/processed/dev_train.data
python3 statistics.py --data ../data/Dialogue/processed/dev_dev.data
python3 statistics.py --data ../data/Dialogue/processed/dev_test.data
```

This will additionally generate the gold-annotation file in the same folder with suffix `.gold`.

## Pre-trained Word Embeddings

The initial model utilized pre-traine PubMed embeddings.

Please download [GloVe embeddings](http://nlp.stanford.edu/data/glove.6B.zip), and put it under [```./embeds```](https://github.com/crystal-xu/KBEnrichment/tree/master/edge-oriented-graph/embeds)

## Train

```
$ cd src/
$ # DocRED
$ python3 eog.py --config ../configs/parameters_docred.yaml --train --gpu 0  
$ # DialogRE
$ python3 eog.py --config ../configs/parameters_dialogue.yaml --train --gpu 0 
```

## Test

```$ python3 eog.py --config ../configs/parameters_docred.yaml --test --gpu 0```

### Post-processing

In order to evaluate the results, the prediction file ```test.preds``` need to be converted to the same format as DocRED:

- DocRED

```
$ # DocRED
$ mkdir ../data/DocRED 
$ # put the test.preds and rel2id.json under the directory
$ python3 convert2DocREDFormat --data DocRED
```

- DialogRE

```
$ mkdir ../data/Dialogue 
$ # put the test.preds and rel2id.json under the directory
$ python3 convert2DocREDFormat --data Dialogue
```



DialogRE
=====

**Main directory**:

[```cd dialogre```](https://github.com/crystal-xu/KBEnrichment/tree/master/dialogre)

**Adapted from**:

https://github.com/nlpdata/dialogre/tree/master

**Reference Paper**:

[Dialogue-Based Relation Extraction](https://arxiv.org/abs/2004.08056)

## **Environment**

Python 3.6 and PyTorch 1.0.

## Preparation

* ```kb/Fandom_triples```: relational triples from [Fandom](https://friends.fandom.com/wiki/Friends_Wiki).
* ```kb/matching_table.txt```: mapping from Fandom relational types to DialogRE relation types.
* ```bert``` folder: a re-implementation of BERT and BERT<sub>S</sub> baselines.
  1. Download and unzip BERT from [here](https://github.com/google-research/bert), and set up the environment variable for BERT by 
     ```export BERT_BASE_DIR=/PATH/TO/BERT/DIR```. 
  2. Copy the dataset folder ```data``` to ```bert/```.
  3. In ```bert```, execute ```python convert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path=$BERT_BASE_DIR/bert_model.ckpt --bert_config_file=$BERT_BASE_DIR/bert_config.json --pytorch_dump_path=$BERT_BASE_DIR/pytorch_model.bin```.

## Train

To run the BERT<sub>S</sub> baseline, execute the following commands in ```bert```:

```
$ cd bert

$ python run_classifier.py   --task_name berts  --do_train --do_eval   --data_dir .   --vocab_file $BERT_BASE_DIR/vocab.txt   --bert_config_file $BERT_BASE_DIR/bert_config.json   --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir berts_f1  --gradient_accumulation_steps 2

$ rm berts_f1/model_best.pt && cp -r berts_f1 berts_f1c && python run_classifier.py   --task_name bertsf1c --do_eval   --data_dir .   --vocab_file $BERT_BASE_DIR/vocab.txt   --bert_config_file $BERT_BASE_DIR/bert_config.json   --init_checkpoint $BERT_BASE_DIR/pytorch_model.bin   --max_seq_length 512   --train_batch_size 24   --learning_rate 3e-5   --num_train_epochs 20.0   --output_dir berts_f1c  --gradient_accumulation_steps 2
```



## Test

To evaluate the BERT<sub>S</sub> baseline, execute the following commands in ```bert```:

```
$ cd bert
$ python evaluate.py --f1dev berts_f1/logits_dev.txt --f1test berts_f1/logits_test.txt --f1cdev berts_f1c/logits_dev.txt --f1ctest berts_f1c/logits_test.txt
```



Evaluations
=====

**Main directory**:

[```cd Evaluation```](https://github.com/crystal-xu/KBEnrichment/tree/master/Evaluation)

Put  ```train_annotated.json```, ```dev.json```, ```test.json``` and prediction results ```dev_test_index.json``` under the directory [```code/DocRED/re_data```](https://github.com/crystal-xu/KBEnrichment/tree/master/Evaluation/code/DocRED) or[ ```code/Dialogue/re_data```](https://github.com/crystal-xu/KBEnrichment/tree/master/Evaluation/code/Dialogue)

- F1-score versus Relation Types

```
$ cd code
$ python3 eval_re_type.py --data DocRED|Dialogue 
```

- [**BERT<sub>S</sub>**](https://github.com/crystal-xu/KBEnrichment/tree/master/dialogre)

```
$ cd ../dialogre/bert
$ python3 evaluate_rel_type.py 
```

- F1-score of Intra- v.s. Inter-sentential Relations

```
$ cd code
$ python3 eval_re_intra_inter.py --data DocRED|Dialogue 
```

- F1-score versus Relation Distances

```
$ cd code
$ python3 eval_re_dist.py --data DocRED|Dialogue
```

- Distributions of Relation Types

```
$ cd code
$ python3 get_re_type_distri.py --data DocRED|Dialogue 
```

- Distributions of  Intra- v.s. Inter-sentential Relations

```
$ cd code
$ python3 get_re_intra_inter_distri.py --data DocRED|Dialogue
```

- Distributions of Relation Distances

```
$ cd code
$ python3 get_re_dist_distri.py --data DocRED|Dialogue
```

- part_of distance distributions

```
$ cd code
$ python3 part_of_birth_distri.py 
```

- date_of_birth distance distributions

```
$ cd code
$ python3 get_date_of_birth_distri.py 
```



# Known Issues

1. A reported bug from the authors of Graph-LSR:  https://github.com/nanguoshun/LSR/issues/9

   Our current workaround:

   - Graph-LSR: change the batch size from 20 to 10.
   - BERT-LSR: change the batch size from 20 to 10; make the document number an integer times the batch size.



# Acknowledgement

We acknowledge that the initial ideas own to the authors of following officially published reference papers and released code.

We also refer to their descriptions of their open source repositories to finish these README file.

## References

[1] [DocRED: A Large-Scale Document-Level Relation Extraction Dataset](https://arxiv.org/abs/1906.06127v3)

[2] [Fine-tune Bert for DocRED with Two-step Process](https://arxiv.org/abs/1909.11898)

[3] [Reasoning with Latent Structure Refinement for Document-Level Relation Extraction](https://arxiv.org/abs/2005.06312)

[4] [Connecting the Dots: Document-level Relation Extraction with Edge-oriented Graphs](https://www.aclweb.org/anthology/D19-1498/)

[5] [Dialogue-Based Relation Extraction](https://arxiv.org/abs/2004.08056)

## Open Source Repositories

[1] https://github.com/thunlp/DocRED/tree/master

[2] https://github.com/hongwang600/DocRed/tree/master

[3] https://github.com/nanguoshun/LSR/tree/master

[4] https://github.com/fenchri/edge-oriented-graph/tree/master

[5] https://github.com/nlpdata/dialogre/tree/master

## 

