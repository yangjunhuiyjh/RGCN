from torch import Tensor, long, relu, cat, randn, tensor, where, zeros, ones, exp, mul, nan_to_num, \
    block_diag
from torch.nn import LeakyReLU, Module, ModuleList
from torch.nn import Parameter, ParameterList
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import dropout_adj


class BlockLinear(Module):
    def __init__(self, in_channels, out_channels, num_blocks) -> None:
        super(BlockLinear, self).__init__()
        self.weights = ParameterList(
            [Parameter(randn(in_channels // num_blocks, out_channels // num_blocks)) for _ in range(num_blocks)])

    def forward(self, x):
        weight = block_diag(*self.weights)
        return x @ weight


class RGCNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relation_types, num_blocks=None, num_bases=None,
                 norm_type='relation-degree', activation=relu, dropout=None):
        """
        Arguments:
            in_channels: dimension of input node feature space
            out_channels: dimension of output node feature space
            num_relation_types: the number of relations for each graph
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
        """
        super(RGCNLayer, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relation_types = num_relation_types
        self.num_blocks = num_blocks
        self.num_bases = num_bases
        self.activation = activation
        self.norm_type = norm_type
        if self.norm_type == 'attention':
            self.leaky_relu = LeakyReLU(0.2)
            self.attention_weights = Parameter(randn(num_relation_types, 2 * out_channels))
            glorot(self.attention_weights)
        if num_blocks is not None:
            assert (in_channels % num_blocks == 0 and out_channels % num_blocks == 0)
            self.weights = ModuleList([BlockLinear(in_channels, out_channels, num_blocks)
                                       for _ in range(num_relation_types)])
            self.block_dim = in_channels // num_blocks
            self.prop_type = 'block'
        elif num_bases is not None:
            # self.basis_vectors = ModuleList([Linear(in_channels,out_channels,False) for _ in range(num_bases)])
            # glorot(self.basis_vectors)
            # self.weights = ParameterList([Parameter(randn(num_bases)) for _ in range(num_relations)])
            self.basis_vectors = Parameter(randn(num_bases, in_channels, out_channels))
            glorot(self.basis_vectors)
            self.weights = Parameter(randn(num_relation_types, num_bases))
            self.prop_type = 'basis'
        else:
            self.weights = ParameterList(
                [Parameter(randn(in_channels, out_channels)) for _ in range(num_relation_types)])
            self.prop_type = None
        self.self_connection = Parameter(randn(in_channels, out_channels))
        glorot(self.weights)
        glorot(self.self_connection)
        self.dropout = dropout

    def partial_message(self, x_l, weight, prop_type):
        if prop_type == 'block':
            return weight(x_l)
        elif prop_type == 'basis':
            return x_l @ (weight @ self.basis_vectors.view(self.num_bases, -1)).view(self.in_channels,
                                                                                     self.out_channels)
        else:
            return x_l @ weight

    def message(self, x_j, x_i, weight_r, norm, prop_type, index, attention=None):
        '''
        x_j is of size [num_neighbours,hidden_dim]
        norm is of size [num_neigbours]
        '''
        if attention is None:
            return norm.view(-1, 1) * self.partial_message(x_j, weight_r, prop_type)
        else:
            message_j = self.partial_message(x_j, weight_r, prop_type)
            message_i = self.partial_message(x_i, weight_r, prop_type)
            a_ij_num = exp(self.leaky_relu(cat([message_i, message_j], dim=-1)) @ attention)
            for i, e in enumerate(index):
                self.r_attention_total[e] += a_ij_num[i]
            return mul(norm, a_ij_num).view(-1, 1) * message_j

    def forward(self, x: Tensor, edge_index, edge_attributes):
        if self.dropout:
            edge_index, edge_attributes = dropout_adj(edge_index, edge_attributes, 2 * self.dropout)
        out = zeros((x.size(0), self.out_channels),device=x.device)
        for r, e in enumerate(self.weights):
            masked_edge_index = edge_index.T[where(edge_attributes[:, r] > 0)].T
            norm = self.compute_norm(edge_index, edge_attributes, r, x)
            if self.norm_type == 'attention':
                self.r_attention_total = zeros(x.size(0), device=x.device)
                messages = self.propagate(masked_edge_index, x=x, weight_r=e, norm=norm, prop_type=self.prop_type,
                                          attention=self.attention_weights[r])
                expanded_divisor = self.r_attention_total.unsqueeze(-1).expand_as(out)
                out += nan_to_num(messages / expanded_divisor, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                out += self.propagate(masked_edge_index, x=x, weight_r=e, norm=norm, prop_type=self.prop_type)
        self_loop_edge_index = tensor([[i, i] for i in range(x.size(0))], dtype=long, device=x.device).T
        if self.dropout:
            self_loop_edge_index, self_loop_edge_attributes = dropout_adj(self_loop_edge_index, p=self.dropout)
        norm = ones(self_loop_edge_index.size(-1),device=x.device)
        out += self.propagate(self_loop_edge_index, x=x, weight_r=self.self_connection, norm=norm, prop_type=None)
        out = self.activation(out)
        return out

    def compute_norm(self, edge_index, edge_attributes, r, x):
        if self.norm_type == 'relation-degree':
            masked_edge_index = edge_index.T[where(edge_attributes[:, r] > 0)].T
            row, col = masked_edge_index
            deg = degree(row, x.size(0))
            norm = nan_to_num(1 / deg[row], nan=0.0, posinf=0.0, neginf=0.0)
        elif self.norm_type == 'non-relation-degree':
            row, col = edge_index
            masked_row, _ = edge_index.T[where(edge_attributes[:, r] > 0)].T
            deg = degree(row, x.size(0))
            norm = nan_to_num(1 / deg[masked_row], nan=0.0, posinf=0.0, neginf=0.0)
        elif self.norm_type == 'attention' or self.norm_type is None:
            masked_edge_index = edge_index.T[where(edge_attributes[:, r] > 0)].T
            row, col = masked_edge_index
            norm = ones(x.size(0), device=edge_index.device)[row]
        else:
            raise ValueError("norm type incorrect", self.norm_type)
        return norm


if __name__ == '__main__':
    # print(RGCNLayer(10,10,100,num_bases=4))
    # print(RGCNConv(10,10,100,num_bases=4))
    ds = TUDataset('/tmp/MUTAG', name='MUTAG')
    print([ds[i] for i in range(4)])
    model = RGCNLayer(7, 3, 4, num_bases=3, norm_type='attention', activation=LeakyReLU())
    dl = DataLoader(ds, batch_size=5)
    t = 0
    for data in dl:
        if t == 0:
            print(model(data.x, data.edge_index, data.edge_attr))
            t = 1
