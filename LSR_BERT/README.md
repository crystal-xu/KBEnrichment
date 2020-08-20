# LSR_BERT
It is the PyTorch implementation of the LSR model with BERT embeddings in ACL 2020 Paper 
"[Reasoning with Latent Structure Refinement for Document-Level Relation Extraction](https://arxiv.org/abs/2005.06312)".

This repository adapted the authors' implementation of LSR with GloVe embeddings: https://github.com/nanguoshun/LSR/tree/master

# Requirement
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
# Dataset

**DocRED**:  please download it [here](https://github.com/thunlp/DocRED/tree/master/data), which are officially provided by [DocRED: A Large-Scale Document-Level Relation Extraction Dataset](https://arxiv.org/abs/1906.06127). 

**DialogRE**: please download it [here](https://github.com/nlpdata/dialogre/tree/master/data), which are officially provided by [Dialogue-Based Relation Extraction](https://arxiv.org/abs/2004.08056). 

# Data Proprocessing
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

# Training

In order to train the model, run:

```
# cd code
# python3 train.py
```

# Test
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


