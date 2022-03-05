from modulefinder import Module
from torch import Tensor, relu, cat, randn, stack, tensor, where, zeros, ones
from torch.nn import ModuleList, Linear, Parameter, ParameterList
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.nn import RGCNConv
from torch_geometric.datasets import TUDataset

class RGCNLayer(MessagePassing):
    def __init__(self,in_channels, out_channels, num_relations, num_blocks=None, num_bases=None):
        '''
        Arguments:
            in_channels: dimension of input node feature space
            out_channels: dimension of output node feature space
            num_relations: the number of relations for each graph
            num_blocks/num_bases if set to not equal None uses alternate scheme for computation

            thus if num_blocks, num_bases set to None ->
                h_i = \sum_{i,r}() TODO
            Note that num_blocks must factor into both in_channels and out_channels
        '''
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_bases = num_bases
        if num_blocks is not None: 
            assert(in_channels%num_blocks==0 and out_channels%num_blocks==0)      
            self.weights = ModuleList([ModuleList([Linear(in_channels//num_blocks,out_channels//num_blocks,False) for _ in range(num_blocks)]) for _ in range(num_relations)])
        elif num_bases is not None:
            self.basis_vectors = ModuleList([Linear(in_channels,out_channels,False) for _ in range(num_bases)])
            self.weights = ParameterList([Parameter(randn(num_bases)) for _ in range(num_relations)])
        else:
            self.weights = ModuleList([Linear(in_channels,out_channels,bias=False) for i in range(num_relations)])
        self.self_connection = Linear(in_channels,out_channels,False)

    def message(self,x_j,weight_r,norm):
        if self.num_blocks is not None:
            norm.view(-1,1) * cat([e(x_j[i*self.num_blocks:(i+1)*self.num_blocks]) for i,e in enumerate(weight_r)])
        elif self.num_bases is not None:
            norm.view(-1,1) * self.weights @ stack([bv(x_j) for bv in self.basis_vectors])
        else:    
            return norm.view(-1,1) * weight_r(x_j)

    def forward(self,x: Tensor, edge_index, edge_attributes):
        out = zeros((x.size(0),self.out_channels))
        for r,e in enumerate(self.weights):
            masked_edge_index = edge_index.T[where(edge_attributes[:,r]>0)].T
            row, col = masked_edge_index
            deg =  degree(col, x.size(0))
            norm = deg[row]
            print(norm)
            out+= self.propagate(masked_edge_index,x=x,weight_r=e,norm=norm)
        self_edge_index = tensor([[i,i] for i in range(x.size(0))],dtype=int).T
        norm = ones(x.size(0))
        out+= self.propagate(self_edge_index,x=x,weight_r=self.self_connection,norm=norm)
        out = relu(out)
        return out
if __name__ == '__main__':
    # print(RGCNLayer(10,10,100,num_bases=4))
    # print(RGCNConv(10,10,100,num_bases=4))
    ds = TUDataset('/tmp/MUTAG',name='MUTAG')
    print(ds[10])
    model = RGCNLayer(7,3,4,num_bases=1)
    data = ds[0]
    print(model(data.x,data.edge_index,data.edge_attr))