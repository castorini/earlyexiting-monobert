# Early Exiting MonoBERT

This is the code base for the paper [Early Exiting BERT for Efficient Document Ranking](https://www.aclweb.org/anthology/2020.sustainlp-1.11/).

## Installation

This repo is tested on Python 3.7.7, PyTorch 1.3.1, and Cuda 10.1. Using a virtualenv or conda environemnt is recommended, for example:

```
conda install pytorch==1.3.1 torchvision cudatoolkit=10.1 -c pytorch
```

Also install the following packages in the environment:

```
tqdm tensorboardX boto3 regex sentencepiece sacremoses scikit-learn pyserini
```

##  Data Preparation

Two datasets are used in this repo: MS MARCO passage and ASNQ. Additionally we can use TREC-DL 2019.

#### MS MARCO passage (https://github.com/microsoft/MSMARCO-Passage-Ranking)

Go to `data/msmarco`, download the training set and extract it:

```
wget https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz
tar -xvzf triples.train.small.tar.gz
```

then extract uniq training data (details can be found in Section 4 of the paper):

```
python convert_data.py triples.train.small.tsv train.uniq.416k.tsv
```

Also in the same folder, download the dev set and extract it:

```
wget https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz
tar -xvzf top1000.dev.tar.gz
```

then partition the dev set (since it's pretty large, it would be easier to run evaluation by partition; there will be 500 queries per partition):

```
python partition_eval.py dev
```

Go to `evaluation/msmarco`, download the qrel collection and extract it (we'll need `qrels.dev.small.tsv` for evaluation):

```
wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar xvzf collectionandqueries.tar.gz
```

#### ASNQ (https://github.com/alexa/wqa_tanda)

Go to `data/asnq`, download the training set and extract it:

```
wget https://wqa-public.s3.amazonaws.com/tanda-aaai-2020/data/asnq.tar
tar xvf asnq.tar
```

then preprocess the dataset and partition the dev set:

```
python preprocess.py
python partition_eval.py dev
```

#### Trec-dl (https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019)

Go to `data/trec-dl`, download and extract the required files into a `raw_data` folder

```
mkdir raw_data
cd raw_data
wget https://trec.nist.gov/data/deep/2019qrels-pass.txt
wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz
tar xzvf msmarco-test2019-queries.tsv.gz
wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz
tar xzvf collection.tar.gz
```

then preprocess the dataset and use BM25 for first stage retrieval (with pyserini):

```
python bm25.py
```

## Training the Model

```
scripts/train.sh bert base DATASET all
```

`bert base` is the pre-trained model; `all` stands for training all layers together.

DATASET can be chosen among `msmarco` or `asnq`. We resue `msmarco`'s trained models for `trec-dl`.

## Evaluating the Model

First go to `evaluation/asnq`, and build the eval tool and link the qrel file over:

```
tar xvzf trec_eval.9.0.4.tar.gz
cd trec_eval.9.0.4
make
cd ..
ln -s ../../data/asnq/asnq.qrel.dev.tsv .
```

Also link the qrel file and trec_eval folder over for trec-dl:

```
# at evaluation/trec-dl
ln -s ../../data/trec-dl/raw_data/2019qrels-pass.txt .
ln -s ../asnq/trec_eval.9.0.4 .
```

#### Evaluate with early exiting

We evaluate the model efficiency with real early exiting.

```
scripts/eval_ee.sh bert base DATASET all PARTITIONS PC NC
```

PARTITIONS is the partitions you wish to evaluate. If you wish to evaluate the entire dev set, it's `0-69` for msmarco, `0-5` for asnq, and `0` for trec-dl.

`PC` and `NC` are positive and negative confidence thresholds.

It generates a folder `evaluation/DATASET/pc-PC-nc-NC`, we can then evaluate it with

```
cd evaluation/DATASET
python direct_eval.py --sp_folder pc-PC-nc-NC
```

please check arguments of `direct_eval.py` for more details. For example, you can specify `-p 1-3` to evaluate only partitions 1, 2, and 3.

#### Evaluate for the purpose of the paper

For more efficient evaluation (using a large number of different thresholds), we can use `eval.sh`. In this way, we actually record scores generated by all layers. Model efficiency will be calculated as average exit layers in later scripts.

```
scripts/eval.sh bert base DATASET all PARTITIONS
```

It generates folders `evaluation/DATASET/layer*`, we can then evaluate them with

```
cd evaluation/DATASET
python direct_eval.py --each_layer  # for evaluating the score of each layer's classifier
python direct_eval.py -pc PC -nc NC  # for evaluating for given positive and negative thresholds
```

