PyTorch implementation of Action-Centric Relation Transformer for VideoQA.

## Steps to run the experiments

### Requirements
Here lists the required packages for our model:
* ``Python 2.7 ``
* ``PyTorch=1.0.1``
* ``tqdm``
* ``nltk<=3.4``
* ``h5py``
* ``pandas``
* ``typing``

You can use the following command to install them:
``pip install h5py nltk==3.3 pandas torch==1.0.1 tqdm typing``


### Datasets and word embeddings
* features: Please download([Baidu Yun](链接: https://pan.baidu.com/s/1zz2dfwsnr4G_QhD8WlBSFQ) with code ``kv8q``. 
* first put all the things with ``features.tar.gz`` into ``data`` directory.
* next combine all ``feature.tar.gz``, extract them into one directory ``features``, put it under the directory ``data``

### Training on TGIF-QA
Model is trained separately on 4 sub-tasks: ``Action,Trans,Frame,Count``, first get into corresponding directory (taking Action as an example):
* cd TGIF_QA/Action

then begin training
* python main.py 

### Training on Anet-QA
* cd Anet-QA
* python main.py
