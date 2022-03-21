from torch import LongTensor, as_tensor, ones, long
from torch.nn.functional import one_hot
from pytorch_lightning import LightningModule
from torch.nn import ModuleList, NLLLoss
from torch import relu, softmax
from torch.optim import Adam
from Models.rgcn import RGCNLayer
class EntityClassificationRGCN(LightningModule):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, num_relations, optimizer= Adam, lr=0.01,**kwargs):
        super(EntityClassificationRGCN,self).__init__()
        self.num_relations = num_relations
        self.layers = ModuleList([RGCNLayer(in_dim,hidden_dim,num_relations,**kwargs)]+[RGCNLayer(hidden_dim,hidden_dim, num_relations,**kwargs) for _ in range(num_layers-2)]+[RGCNLayer(hidden_dim,out_dim, num_relations)])
        self.loss = NLLLoss()
        self.optimizer = optimizer
        self.lr = lr

    def forward(self, x, edge_index, edge_attributes):
        for l in self.layers:
            x = l(x, edge_index, edge_attributes)
        return softmax(x,-1)

    def training_step(self, batch, batch_idx):
        print(batch)
        x, edge_index, edge_attributes, y = batch.x, batch.edge_index, batch.edge_type, batch.train_y
        edge_attributes = one_hot(edge_attributes,num_classes=self.num_relations)
        if x is None:##If there are no initial features, just use a one to encode it
            x = one_hot(as_tensor([i for i in range(batch.num_nodes)],dtype=long)).float()
        print(x)
        for l in self.layers:
            x = l(x, edge_index, edge_attributes)
        #### CODE UP TO HERE IS KINDA NASTY -- SHOULD WORK ON MAKING DATALOADERS MORE STANDARDISED...
        x = softmax(x[batch.train_idx],-1)
        loss = self.loss(x,y)
        self.log("train_loss",loss.item())
        return loss

    def configure_optimizers(self):
        return self.optimizer(self.parameters(),lr=self.lr)