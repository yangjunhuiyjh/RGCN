import enum
from modulefinder import Module
from numpy import zeros
from torch import Tensor, relu, cat
from torch.nn import ModuleList, Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

class RGCNLayer(MessagePassing):
    def __init__(self,in_channels, out_channels, num_relations, num_blocks=None, num_bases=None):
        '''
        Arguments:
            in_channels: dimension of input node feature space
            out_channels: dimension of output node feature space
            num_relations: the number of relations for each graph
            num_blocks/num_bases if set to not equal None uses alternate scheme for computation

            thus if num_blocks, num_bases set to None ->
                h_i = \sum_{i,r}()
            Note that num_blocks must factor into both in_channels and out_channels
        '''
        super().__init__(aggr='add')
        self.num_blocks = num_blocks
        self.num_bases = num_bases
        if num_blocks is not None:       
            self.weights = ModuleList([ModuleList([Linear(in_channels//num_blocks,out_channels//num_blocks,False) for _ in range(num_blocks)]) for _ in range(num_relations)])
        elif num_bases is not None:
            self.weights = ...
        else:
            self.weights = ModuleList([Linear(in_channels,out_channels,bias=False) for i in range(num_relations)])
            self.self_connection = Linear(in_channels,out_channels,False)

    def message(self,x_j,weight_r,norm):
        if self.num_blocks is not None:
            norm.view(-1,1) * cat([e(x_j[i*self.num_blocks:(i+1)*self.num_blocks]) for i,e in enumerate(weight_r)])
        elif self.num_bases is not None:
            pass
        else:    
            return norm.view(-1,1) * weight_r(x_j)

    def forward(self,x: Tensor,edge_index):

        out = zeros((x.size(0),self.out_channels))
        for r,e in enumerate(self.weights):
            row, col = edge_index[r]
            deg =  degree(col, x.size(0))
            norm = deg[row]
            out+= self.propagate(edge_index[r],x=x,weight_r=e,norm=norm)
        out+= self.propagate()
        out = relu(out)
        return out
if __name__ == '__main__':
    print(RGCNLayer(10,10,100))