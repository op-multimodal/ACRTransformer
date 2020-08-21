PyTorch implementation of Action-Centric Relation Transformer for VideoQA.

## Steps to run the experiments

### Requirements
* ``Python 2.7 ``
* ``PyTorch=1.0.1``
* ``tqdm``
* ``nltk<=3.4``
* ``h5py``
* ``pandas``
* ``typing``

``pip install h5py nltk==3.3 pandas torch==1.0.1 tqdm typing``


### Datasets and word embeddings
* features: Please download([Baidu Yun](https://pan.baidu.com/s/1rvhmrl36KCfHYCnggHYl5Q) with code ``be85`` and put it into ``data`` directory.

### Training on TGIF-QA
Model is trained separately on 4 sub-tasks: ``Action,Trans,Frame,Count``, first get into corresponding directory (taking Action as an example):
* cd TGIF_QA/Action
then begin training
* python main.py 

### Training on Anet-QA
* cd Anet-QA
* python main.py
