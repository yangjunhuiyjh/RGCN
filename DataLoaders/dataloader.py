from torch_geometric.datasets import RelLinkPredDataset, WordNet18, Entities
from dgl import to_networkx
from dgl.data import FB15kDataset
import torch
from torch_geometric.utils.convert import from_networkx
import os

# Valid names are:
# fb15k237, wn18, fb15k, mutag, aifb, bgs, am


def get_dataset(name: str):
    dirname = os.path.dirname(__file__)
    return [torch.load(os.path.join(dirname,name+".pt"))]