from torch_geometric.datasets import RelLinkPredDataset


if __name__=='__main__':
    ds = RelLinkPredDataset("Datasets/fb15k237","FB15k-237")
    print(ds[0])