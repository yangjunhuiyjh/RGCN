from numpy import expand_dims
from torch.nn import Module, Parameter
from torch import tensor, randn, load, sum, stack
from torch_geometric.loader import DataLoader
import os

class LinkFeat(Module):
    def __init__(self, num_nodes, num_relation_types):
        super(LinkFeat,self).__init__()
        self.num_nodes = num_nodes
        self.num_relation_types = num_relation_types
        self.edgeparam = Parameter(randn(num_relation_types))
        self.subjparam = Parameter(randn(num_relation_types, num_nodes))
        self.objparam = Parameter(randn(num_relation_types, num_nodes))

    def forward(self, edge_index, edge_type):
        return (edge_index, edge_type)

    def score(self, s, r, o, x):
        edge_index, edge_type = x
        mask = edge_type.repeat(s.size(-1)).view(s.size(-1),-1)[edge_index[0].eq(s.unsqueeze(-1))*edge_index[1].eq(o.unsqueeze(-1))] ## mask should be dimension (batch_size, num_relation_types)
        edge_score = self.edgeparam[mask]-self.edgeparam[r]
        a = self.objparam
        b = o
        print(self.objparam[r].shape)
        print(o.shape)
        return stack([edge_score,self.objparam[r][o.unsqueeze(-1)],self.subjparam[r][s.unsqueeze(-1)]]).sum()


if __name__ == '__main__':
    ds = load(os.path.join("/Users/dylanhillier/Oxford/RGCN/DataLoaders","wn18.pt"))
    model = LinkFeat(40943,36)
    dl = DataLoader([ds], batch_size=1)
    t = 0
    for data in dl:
        if t == 0:
            print(data)
            edge_index, edge_type = model(data.edge_index, data.edge_type)
            print(model.score(edge_index[0][:100],edge_type[:100], edge_index[1][:100],(edge_index,edge_type)))
            t = 1