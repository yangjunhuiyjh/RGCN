from torch_geometric.datasets import RelLinkPredDataset, WordNet18, Entities
from dgl import to_networkx
from dgl.data import FB15kDataset
from torch_geometric.utils.convert import from_networkx


def get_dataset(name:str):
    if name == 'fb15k237':
        return RelLinkPredDataset("Datasets/fb15k237","FB15k-237")[0]
    elif name == 'wn18':
        return WordNet18("Datasets/wn18")[0]
    elif name =="mutag":
        return Entities("Datasets/mutag","MUTAG")[0]
    elif name == 'fb15k':
        ds = FB15kDataset(raw_dir="Datasets/fb15k")
        ds_networkx = to_networkx(ds[0])
        return from_networkx(ds_networkx)
    elif name == 'aifb':
        return Entities("Datasets/aifb","AIFB")[0]
    elif name == 'bgs':
        return Entities("Datasets/bgs","BGS")[0]
    elif name == 'am':
        return Entities("Datasets/am", "AM")[0]