from torch.nn.parameter import Parameter
from torch import FloatTensor, LongTensor, as_tensor, norm, ones, long
from torch.nn.functional import one_hot
from pytorch_lightning import LightningModule
from torch.nn import ModuleList, CrossEntropyLoss
from torch import relu, softmax
from torch.optim import Adam
from Models.rgcn import RGCNLayer
from Models.DistMult import distMult
from torch.nn.init import kaiming_normal_
from torch_geometric.utils import negative_sampling

class LinkPredictionRGCN(LightningModule):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, num_relations, num_entities, omega=1, l2lambda=0.01, optimizer= Adam, lr=0.01,**kwargs):
        super(LinkPredictionRGCN,self).__init__()
        self.num_relations = num_relations
        self.embedder = Parameter(FloatTensor(num_entities,in_dim))
        kaiming_normal_(self.embedder)
        self.layers = ModuleList([RGCNLayer(in_dim,hidden_dim,num_relations,**kwargs)]+[RGCNLayer(hidden_dim,hidden_dim, num_relations,**kwargs) for _ in range(num_layers-2)]+[RGCNLayer(hidden_dim,out_dim, num_relations)])
        self.loss = CrossEntropyLoss(reduction='sum')
        self.distmult = distMult(out_dim,num_relations)
        self.optimizer = optimizer
        self.omega = omega
        self.num_entities = num_entities
        self.lr = lr
        self.l2lambda = l2lambda
        self.save_hyperparameters()

    def forward(self, edge_index, edge_types):
        x = self.embedder
        edge_attributes = one_hot(edge_types,num_classes=self.num_relations)
        for l in self.layers:
            x = l(x, edge_index, edge_attributes)
        return x

    def training_step(self, batch, batch_idx):
        print(batch)
        edge_index = batch.train_edge_index
        edge_attributes = batch.train_edge_type
        edge_attributes = one_hot(edge_attributes,num_classes=self.num_relations)
        x = self.embedder
        for l in self.layers:
            x = l(x, edge_index, edge_attributes)
        #### CODE UP TO HERE IS KINDA NASTY -- SHOULD WORK ON MAKING DATALOADERS MORE STANDARDISED...
        loss = 0
        print("done with embedding")
        for _ in range(self.omega):
            edges = negative_sampling(edge_index,self.num_entities)
            print("done sampling")
            score = self.distmult(x[edges[0]],batch.edge_type,x[edges[1]]) ### Correct this
            loss += self.loss(score,0)
        print("done with neg samples")
        edges = edge_attributes
        score = self.distmult(x[edges[0]],edge_attributes[edges[0]],x[edges[1]])
        loss += self.loss(score,1)/(edge_attributes.size(1)*(1+self.omega))
        for name, param in self.distmult.named_parameters(): ## L2 Loss
            loss += norm(param) * self.l2lambda
        self.log("train_loss",loss.item())
        return loss

    def score(self, s, p, o, x):
        '''
        s: index of subject
        p: index of relation
        o: index of object
        '''
        score = self.distmult(x[s].unsqueeze(0),p.unsqueeze(0),x[o].unsqueeze(0))
        return score

    def configure_optimizers(self):
        return self.optimizer(self.parameters(),lr=self.lr)