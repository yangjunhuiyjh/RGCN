from torch_geometric.datasets import RelLinkPredDataset, WordNet18, Entities
from dgl import to_networkx
from dgl.data import FB15kDataset
import torch
from torch_geometric.utils.convert import from_networkx


def get_dataset(name: str):
    if name == 'fb15k237':
        return RelLinkPredDataset("Datasets/fb15k237", "FB15k-237")
    elif name == 'wn18':
        data1 = WordNet18("Datasets/wn18")[0]
        train_row = data1.edge_index[0][torch.nonzero(data1.train_mask)]
        train_col = data1.edge_index[1][torch.nonzero(data1.train_mask)]
        train_row = train_row.numpy().flatten()
        train_col = train_col.numpy().flatten()
        data1.train_edge_index = torch.tensor([train_row,train_col])
    
        test_row = data1.edge_index[0][torch.nonzero(data1.test_mask)]
        test_col = data1.edge_index[1][torch.nonzero(data1.test_mask)]
        test_row = test_row.numpy().flatten()
        test_col = test_col.numpy().flatten()
    
        valid_row = data1.edge_index[0][torch.nonzero(data1.val_mask)]
        valid_col = data1.edge_index[1][torch.nonzero(data1.val_mask)]
        valid_row = valid_row.numpy().flatten()
        valid_col = valid_col.numpy().flatten()
    
        data1.test_edge_index = torch.tensor([test_row,test_col])
        data1.train_edge_index = torch.tensor([train_row, train_col])
        data1.valid_edge_index = torch.tensor([valid_row, valid_col])
        data1.train_edge_type = torch.flatten(data1.edge_type[torch.nonzero(data1.train_mask)])
        data1.test_edge_type = torch.flatten(data1.edge_type[torch.nonzero(data1.test_mask)])
        data1.valid_edge_type = torch.flatten(data1.edge_type[torch.nonzero(data1.val_mask)])
        return [data1]
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
