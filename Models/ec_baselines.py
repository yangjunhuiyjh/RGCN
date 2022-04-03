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


class ec_baseline(Module):
    def __init__(self,num_nodes,num_relation_types) -> None:
        self.x = zeros((num_nodes,num_relation_types))
        self.num_relation_types = num_relation_types
        self.num_nodes = num_nodes

    def generate_feat_features(self,edge_index: Tensor, edge_type: Tensor):
        for i,edge in tqdm.tqdm(enumerate(edge_index.T)):
            u = edge[0].item()
            r = edge_type[i].item()
            self.x[u,r]+=1

    def generate_wl_features(self,ids,edge_index: Tensor, edge_type: Tensor):
        graphs = []
        for id in ids:
            nodes,ec,mapping,em = k_hop_subgraph(id,100,edge_index,True)
            g = igraph.Graph()
            g.add_vertices(nodes.size(0))
            g.add_edges(ec.T)
            graphs.append(g)
        k_wl = gk.CaculateWLKernel(graphs, par =4)
        return k_wl
        

    def run_svm(self,c,train_idx,train_y):
        svc = svm.LinearSVC(C=c)
        svc.fit(self.x[train_idx],train_y)
        return svc
    
    def grid_search(self,train_idx,train_y):
        svc = GridSearchCV(
        svm.SVC(), {"C": [10 ** i for i in range(-4, 4)]}
        )
        svc.fit(self.x[train_idx],train_y)
        return svc

if __name__ == '__main__':
    ds = Entities("Datasets/mutag","mutag")
    print(ds[0])
    ds[0]
    ids = [i.item() for i in ds[0].train_idx]
    ids += [i.item() for i in ds[0].test_idx]
    model = ec_baseline(23644,46)
    model.generate_wl_features(ds[0].edge_index,None)
    model.generate_feat_features(ds[0].edge_index,ds[0].edge_type)
    svc = model.grid_search(ds[0].train_idx,ds[0].train_y)
    pred_y = svc.predict(model.x[ds[0].test_idx])
    y = ds[0].test_y.numpy()
    correct = 0
    total = 0
    for i,e in enumerate(pred_y):
        if y[i]==e:
            correct+=1
        total+=1

    print(correct/total)
        
