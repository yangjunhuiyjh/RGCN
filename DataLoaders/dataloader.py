from torch_geometric.datasets import RelLinkPredDataset, WordNet18, Entities
from dgl import to_networkx
from dgl.data import FB15kDataset
from torch_geometric.utils.convert import from_networkx


def get_dataset(name: str):
    if name == 'fb15k237':
        return RelLinkPredDataset("Datasets/fb15k237", "FB15k-237")
    elif name == 'wn18':
        return WordNet18("Datasets/wn18")
    elif name == "mutag":
        return Entities("Datasets/mutag", "MUTAG")
    elif name == 'fb15k':
        ds = FB15kDataset(raw_dir="Datasets/fb15k")[0]
        # items = [from_networkx(to_networkx(item, node_attrs= item.ndata.keys(), edge_attrs=item.edata.keys())) for item in ds]
        ds_networkx = to_networkx(ds, node_attrs=ds.ndata.keys(), edge_attrs=ds.edata.keys())
        return [from_networkx(ds_networkx)]
        # return items
    elif name == 'aifb':
        return Entities("Datasets/aifb", "AIFB")
    elif name == 'bgs':
        return Entities("Datasets/bgs", "BGS")
    elif name == 'am':
        return Entities("Datasets/am", "AM")
    else:
        raise ValueError(name)
