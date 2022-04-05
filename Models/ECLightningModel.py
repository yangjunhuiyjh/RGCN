from torch import as_tensor, norm, long
from torch.nn.functional import one_hot
from pytorch_lightning import LightningModule
from torch.nn import ModuleList, CrossEntropyLoss, Linear
from torch import relu, softmax, max
from torch.optim import Adam
import torchmetrics
from Models.rgcn import RGCNLayer
from torch_geometric.nn import RGCNConv


class EntityClassificationRGCN(LightningModule):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, num_relations, l2lambda=0.01, optimizer=Adam, lr=0.01,
                 simplified=False, **kwargs):
        super(EntityClassificationRGCN, self).__init__()
        self.num_relations = num_relations
        self.simplified = simplified
        # self.layers = ModuleList([RGCNConv(in_dim,hidden_dim,num_relations,bias=False,root_weight=True,aggr='add')]+[RGCNConv(hidden_dim,hidden_dim,num_relations,bias=False,root_weight=True,aggr='add') for _ in range(num_layers-2)]+[RGCNConv(hidden_dim,out_dim,num_relations,bias=False,root_weight=True,aggr='add')])
        if simplified:
            self.encoder = Linear(in_dim, hidden_dim)
            self.layers = ModuleList(
                [RGCNLayer(hidden_dim, hidden_dim, num_relations, **kwargs) for _ in range(num_layers - 1)] + [
                    RGCNLayer(hidden_dim, out_dim, num_relations, activation=lambda x: x, **kwargs)])
        else:
            self.layers = ModuleList([RGCNLayer(in_dim, hidden_dim, num_relations, **kwargs)] + [
                RGCNLayer(hidden_dim, hidden_dim, num_relations, **kwargs) for _ in range(num_layers - 2)] + [
                                         RGCNLayer(hidden_dim, out_dim, num_relations, activation=lambda x: x,
                                                   **kwargs)])
        self.loss = CrossEntropyLoss(reduction='sum')
        self.optimizer = optimizer
        self.lr = lr
        self.l2lambda = l2lambda
        self.save_hyperparameters()
        self.train_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

    def forward(self, x, edge_index, edge_attributes):
        if self.simplified:
            x = self.encoder(x)
        for l in self.layers:
            x = l(x, edge_index, edge_attributes)
        return softmax(x, -1)

    def training_step(self, batch, batch_idx):
        x, edge_index, edge_attributes, y = batch.x, batch.edge_index, batch.edge_type, batch.train_y
        edge_attributes = one_hot(edge_attributes, num_classes=self.num_relations)
        if x is None:  # If there are no initial features, just use a one to encode it
            x = one_hot(as_tensor([i for i in range(batch.num_nodes)], dtype=long)).float()
        if self.simplified:
            x = self.encoder(x)
        for l in self.layers:
            x = l(x, edge_index, edge_attributes)
        # CODE UP TO HERE IS KINDA NASTY -- SHOULD WORK ON MAKING DATALOADERS MORE STANDARDISED...
        x = softmax(x[batch.train_idx], -1)
        loss = self.loss(x, y)
        for name, param in self.layers[0].named_parameters():  # L2 Loss
            loss += norm(param) * self.l2lambda
        self.log("train_loss", loss.item())
        y_pred = max(x, -1).indices
        self.train_accuracy(y_pred, y)
        self.log("train_acc", self.train_accuracy)
        return loss

    def train_step_end(self, outs):
        self.log("train_epoch_acc", self.train_accuracy)

    def test_step(self, batch, batch_idx):
        x, edge_index, edge_attributes, y = batch.x, batch.edge_index, batch.edge_type, batch.test_y
        edge_attributes = one_hot(edge_attributes, num_classes=self.num_relations)
        if x is None:  # If there are no initial features, just use a one to encode it
            x = one_hot(as_tensor([i for i in range(batch.num_nodes)], dtype=long)).float()
        if self.simplified:
            x = self.encoder(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attributes)
        # CODE UP TO HERE IS KINDA NASTY -- SHOULD WORK ON MAKING DATALOADERS MORE STANDARDISED...
        x = softmax(x[batch.test_idx], -1)
        y_pred = max(x, -1, ).indices
        print(x, y_pred, y)
        self.test_accuracy(y_pred, y)
        self.log("test_acc", self.test_accuracy)

    def test_step_end(self, outs):
        self.log("test_epoch_acc", self.test_accuracy)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)
