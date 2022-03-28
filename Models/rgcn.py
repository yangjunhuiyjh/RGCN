from torch import FloatTensor, Tensor, relu, cat, randn, stack, tensor, where, zeros, ones, exp, mul, nan_to_num, block_diag
from torch.nn import LeakyReLU
from torch.nn import ModuleList, Linear, Parameter, ParameterList
from torch_geometric.nn import MessagePassing, GATv2Conv
from torch_geometric.utils import degree
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.inits import glorot
'''
TODO: 
> Set Initialisation Scheme
> Test!
> Add attention scheme
'''
class RGCNLayer(MessagePassing):
    def __init__(self,in_channels, out_channels, num_relations, num_blocks=None, num_bases=None, norm_type='relation-degree',activation=relu):
        '''
        Arguments:
            in_channels: dimension of input node feature space
            out_channels: dimension of output node feature space
            num_relations: the number of relations for each graph
            num_blocks/num_bases if set to not equal None uses alternate scheme for computation
            norm_type: str in {"relation-degree","non-relation-degree","attention"}, which normalisation schema to use
            activation: sets which activation function to use (from torch.functional)

            thus if num_blocks, num_bases set to None ->
                This uses equation (2) in the paper
            if num_blocks set to k ->
                This uses equation (3) in the paper
            if num_bases set to k ->
                This uses equation (4) in the paper
            Note that num_blocks must factor into both in_channels and out_channels
        '''
        super(RGCNLayer,self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_blocks = num_blocks
        self.num_bases = num_bases
        self.activation = activation
        self.norm_type = norm_type
        if self.norm_type == 'attention':
            self.leaky_relu = LeakyReLU()
            self.attention_weights = ParameterList([Parameter(randn(2*out_channels)) for _ in range(num_relations)])
            # glorot(self.attention_weights)
        if num_blocks is not None: 
            assert(in_channels%num_blocks==0 and out_channels%num_blocks==0)      
            self.weights = ParameterList([Parameter(block_diag(*[randn(in_channels//num_blocks,out_channels//num_blocks) for _ in range(num_blocks)])) for _ in range(num_relations)])
            self.block_dim = in_channels//num_blocks
            self.prop_type='block'
        elif num_bases is not None:
            self.basis_vectors = ModuleList([Linear(in_channels,out_channels,False) for _ in range(num_bases)])
            glorot(self.basis_vectors)
            self.weights = ParameterList([Parameter(randn(num_bases)) for _ in range(num_relations)])
            self.prop_type='basis'
        else:
            self.weights = ParameterList([Parameter(randn(in_channels,out_channels)) for _ in range(num_relations)])
            self.prop_type=None
        self.self_connection = Parameter(randn(in_channels,out_channels))
        glorot(self.weights)
        glorot(self.self_connection)

    def partial_message(self,x_l,weight,prop_type):
        if prop_type=='block':
            # message= cat([e(x_l[:,i*self.block_dim:(i+1)*self.block_dim]) for i,e in enumerate(weight)],dim=-1)
            message = x_l @ weight
            return message
        elif prop_type=='basis':
            return (stack([bv(x_l) for bv in self.basis_vectors],-1) @ weight)
        else:
            return x_l @ weight

    def message(self,x_j,x_i,weight_r,norm,prop_type,index,attention=None):
        if attention is None:
            return norm.view(-1,1) * self.partial_message(x_j,weight_r,prop_type)
        else:
            message_j = self.partial_message(x_j,weight_r,prop_type)
            message_i = self.partial_message(x_i,weight_r,prop_type)
            a_ij_num = exp(self.leaky_relu(cat([message_i,message_j],dim=-1))@attention)
            self.r_attention_total[index]+=a_ij_num
            return mul(norm,a_ij_num).view(-1,1) * message_j

    def forward(self,x: Tensor, edge_index, edge_attributes):
        out = zeros((x.size(0),self.out_channels))
        for r,e in enumerate(self.weights):
            masked_edge_index = edge_index.T[where(edge_attributes[:,r]>0)].T
            norm = self.compute_norm(edge_index,edge_attributes,r,x)
            if self.norm_type == 'attention':
                self.r_attention_total = zeros(x.size(0))
                messages = self.propagate(masked_edge_index,x=x,weight_r=e,norm=norm,prop_type=self.prop_type,attention=self.attention_weights[r])
                expanded_divisor = self.r_attention_total.unsqueeze(-1).expand_as(out)
                out+=nan_to_num(messages/expanded_divisor,nan=0.0,posinf=0.0,neginf=0.0)
            else:
                out+= self.propagate(masked_edge_index,x=x,weight_r=e,norm=norm,prop_type=self.prop_type)
        self_edge_index = tensor([[i,i] for i in range(x.size(0))],dtype=int).T
        norm = ones(x.size(0))
        out+= self.propagate(self_edge_index,x=x,weight_r=self.self_connection,norm=norm,prop_type=None)
        out = self.activation(out)
        return out

    def compute_norm(self,edge_index,edge_attributes,r,x):
        if self.norm_type =='relation-degree':
            masked_edge_index = edge_index.T[where(edge_attributes[:,r]>0)].T
            row, col = masked_edge_index
            deg =  degree(row, x.size(0))
            norm = nan_to_num(1/deg[row],nan=0.0,posinf=0.0,neginf=0.0)
        elif self.norm_type =='non-relation-degree':
            # deg = []
            # masked_edge_index = edge_index.T[where(edge_attributes[:,r]>0)].T
            # r_row, _ = masked_edge_index
            # for r in range(self.num_relations):
            #     masked_edge_index = edge_index.T[where(edge_attributes[:,r]>0)].T
            #     row, col = masked_edge_index
            #     deg.append(degree(col, x.size(0)))
            # deg = stack(deg,0).sum(0)
            # norm = nan_to_num(1/deg[r_row],nan=0.0,posinf=0.0,neginf=0.0)
            row, col = edge_index
            deg =  degree(row, x.size(0))
            norm = nan_to_num(1/deg[row],nan=0.0,posinf=0.0,neginf=0.0)
        elif self.norm_type =='attention' or self.norm_type == None:
            masked_edge_index = edge_index.T[where(edge_attributes[:,r]>0)].T
            row, col = masked_edge_index
            norm = ones(x.size(0))[row]
        return norm

if __name__ == '__main__':
    # print(RGCNLayer(10,10,100,num_bases=4))
    # print(RGCNConv(10,10,100,num_bases=4))
    ds = TUDataset('/tmp/MUTAG',name='MUTAG')
    print([ds[i] for i in range(4)])
    model = RGCNLayer(7,3,4,num_bases=3,norm_type='attention',activation=LeakyReLU())
    dl = DataLoader(ds,batch_size=5)
    t =0
    for data in dl:
        if t == 0:
            print(model(data.x,data.edge_index,data.edge_attr))
            t=1
