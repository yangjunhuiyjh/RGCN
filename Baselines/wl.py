from sklearn.model_selection import cross_val_predict
from torch.nn import Module
from numpy import zeros
from torch import Tensor, argmax
from torch_geometric.datasets import Entities
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import tqdm
from torch_geometric.utils import k_hop_subgraph
import igraph
import graphkernels.kernels as gk


class WL(Module):
    def __init__(self, train_idx, test_idx) -> None:
        self.size = train_idx.size(0) + test_idx.size(0)
        self.train_set_size = train_idx.size(0)
        self.x = zeros((self.size, self.size))

    def generate_wl_features(self, ids, edge_index: Tensor):
        graphs = []
        for id in tqdm.tqdm(ids):
            nodes, ec, _, _ = k_hop_subgraph(id, 100, edge_index, True)
            g = igraph.Graph()
            g.add_vertices(nodes.size(0))
            edges = [(par[0].item(), par[1].item()) for par in ec.T]
            g.add_edges(edges)
            graphs.append(g)
        k_wl = gk.CalculateWLKernel(graphs, par=4)
        print("features generated")
        self.x = k_wl

        return k_wl
    
    def grid_search(self, train_y):
        svc = GridSearchCV(
        svm.SVC(), {"C": [10 ** i for i in range(-4, 4)]}
        )
        print("fitting")
        svc.fit(self.x[:self.train_set_size], train_y)
        return svc

    def predict(self, svc, test_y):
        pred_y = svc.predict(self.x[self.train_set_size:])
        correct = 0
        total = 0
        for i, e in enumerate(pred_y):
            if test_y[i].item() == e:
                correct += 1
            total += 1

        return correct / total


if __name__ == '__main__':
    ds = Entities("../Datasets/mutag", "mutag")
    print(ds[0])

    ids = [i.item() for i in ds[0].train_idx]
    ids += [i.item() for i in ds[0].test_idx]
    model = WL(ds[0].train_idx, ds[0].test_idx)
    model.generate_wl_features(ids, ds[0].edge_index)
    svc = model.grid_search(ds[0].train_y)
    print(model.predict(svc, ds[0].test_y.numpy()))
