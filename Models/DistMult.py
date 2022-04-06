import torch
from torch import nn


class distMult(torch.nn.Module):
    def __init__(self, dim, num_relation_types, init_mean=0.0, init_std=1.0):
        super(distMult, self).__init__()
        self.dim = dim
        self.diag = nn.parameter.Parameter(
            torch.normal(mean=init_mean, std=init_std, size=(num_relation_types, self.dim)))

    def forward(self, h, r, t):
        """
        batched queries (h,r,t):
        h: (batch_size,dim)
        r: (batch_size,)
        t: (batch_size,dim)

        return:
           output: (batch_size), scores for queries
        """
        return torch.sum(h * t * self.diag[r, :], dim=-1)


if __name__ == '__main__':
    model = distMult(5, 5)
    h = torch.randn(4, 5)
    t = torch.randn(4, 5)
    r = torch.tensor([1, 1, 4, 3], dtype=torch.long)
    print(model(h, r, t))
