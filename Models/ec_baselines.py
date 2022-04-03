from sklearn.model_selection import cross_val_predict
from torch.nn import Module
from numpy import zeros
from torch import Tensor, argmax
from torch_geometric.datasets import Entities
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import tqdm


class ec_baseline(Module):
    def __init__(self,num_nodes,num_relation_types) -> None:
        self.x = zeros((num_nodes,num_relation_types))
        self.num_relation_types = num_relation_types

    def generate_feat_features(self,edge_index: Tensor, edge_type: Tensor):
        for i,edge in tqdm.tqdm(enumerate(edge_index.T)):
            u = edge[0].item()
            r = edge_type[i].item()
            self.x[u,r]+=1

    def generate_wl_features(self,edge_index: Tensor, edge_type: Tensor):
        pass

    def run_svm(self,c,train_idx,train_y):
        svc = svm.LinearSVC(C=c)
        svc.fit(self.x[train_idx],train_y)
        return svc
    
    def grid_search(self,train_idx,train_y):
        svc = GridSearchCV(
        svm.SVC(), {"C": [10 ** i for i in range(-3, 4)]}
        )
        svc.fit(self.x[train_idx],train_y)
        return svc

if __name__ == '__main__':
    ds = Entities("Datasets/aifb","aifb")
    print(ds[0])
    model = ec_baseline(8285,90)
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
        
