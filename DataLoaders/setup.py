from torch_geometric.datasets import RelLinkPredDataset, WordNet18, Entities
from dgl import to_networkx
from dgl.data import FB15kDataset
import torch
from torch_geometric.utils.convert import from_networkx
import os


def download_dataset(name: str):
    dirname = os.path.dirname(__file__)
    if name == 'fb15k237':
        torch.save(RelLinkPredDataset("Datasets/fb15k237", "FB15k-237")[0],os.path.join(dirname,name+".pt"))
    elif name == 'wn18':
        data1 = WordNet18("Datasets/wn18")[0]
        train_row = data1.edge_index[0][torch.nonzero(data1.train_mask)]
        train_col = data1.edge_index[1][torch.nonzero(data1.train_mask)]
        train_row = train_row.numpy().flatten()
        train_col = train_col.numpy().flatten()
    
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
        torch.save(data1, os.path.join(dirname,name+".pt"))
    elif name == "mutag":
        torch.save(Entities("Datasets/mutag", "MUTAG")[0], os.path.join(dirname,name+".pt"))
    elif name == 'fb15k':
        ds = FB15kDataset(raw_dir="Datasets/fb15k")[0]
        # items = [from_networkx(to_networkx(item, node_attrs= item.ndata.keys(), edge_attrs=item.edata.keys())) for item in ds]
        ds_networkx = to_networkx(ds, node_attrs=ds.ndata.keys(), edge_attrs=ds.edata.keys())
        data1 = from_networkx(ds_networkx)
        data1.edge_type = data1.etype

        train_row = data1.edge_index[0][torch.nonzero(data1.train_mask)]
        train_col = data1.edge_index[1][torch.nonzero(data1.train_mask)]
        train_row = train_row.numpy().flatten()
        train_col = train_col.numpy().flatten()
    
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
        torch.save(data1,os.path.join(dirname,name+".pt"))
        # return items
    elif name == 'aifb':
        torch.save(Entities("Datasets/aifb", "AIFB")[0],os.path.join(dirname,name+".pt"))
    elif name == 'bgs':
        torch.save(Entities("Datasets/bgs", "BGS")[0],os.path.join(dirname,name+".pt"))
    elif name == 'am':
        torch.save(Entities("Datasets/am", "AM")[0],os.path.join(dirname,name+".pt"))


if __name__ == '__main__':
    download_dataset('fb15k237')
    download_dataset('wn18')
    download_dataset('fb15k')
    download_dataset('mutag')
    download_dataset('aifb')
    download_dataset('bgs')
    download_dataset('am')