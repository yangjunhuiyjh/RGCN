import torch
from torch import nn


class distMult(torch.nn.Module):
    def __init__(self, dim, num_relations, init_mean=0.0, init_std=1.0):
        super(distMult, self).__init__()
        self.dim = dim
        self.num_relations = num_relations
        self.diag = nn.parameter.Parameter(
            torch.normal(mean=init_mean, std=init_std, size=(self.num_relations, self.dim)))

    def forward(self, h, r, t):
        '''
        batched queries (h,r,t):
        h: (batch_size,dim)
        r: (batch_size,)
        t: (batch_size,dim)

        return:
           output: (batch_size), scores for queries 
      '''
        if len(h.shape) == 1:
            batch_size = 1
        else:
            batch_size = h.shape[0]
        return torch.sum(h * t * self.diag[r, :], dim=1)


if __name__ == '__main__':
    model = distMult(5, 5)
    h = torch.randn(4, 5)
    t = torch.randn(4, 5)
    r = torch.tensor([1, 1, 4, 3], dtype=torch.long)
    print(model(h, r, t))
