# RGCN
Implementation of RGCN model and experiments from the paper [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/pdf/1703.06103.pdf) by Schlichtkrull, Kipf et al. 2017
## Install
run 
```shell
$ pip install -r requirements.txt
```
## Setup
To install a dataset run the dataloader file relating to the dataset. Thus first run:
```shell
$ python DataLoaders/setup.py
```

## Run experiments
nohup bash Scripts/xxx.sh &>xxx &