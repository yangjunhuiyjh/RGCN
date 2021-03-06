from base64 import encode
from torch import as_tensor, norm, long, zeros
from torch.nn.functional import one_hot
from pytorch_lightning import LightningModule
from torch.nn import ModuleList, CrossEntropyLoss, Linear
from torch import relu, softmax, max
from torch.optim import Adam
import torchmetrics
from Models.rgcn import RGCNLayer

def generate_feat_features(x,edge_index, edge_type):
    for i,edge in enumerate(edge_index.T):
        u = edge[0].item()
        r = edge_type[i].item()
        x[u,r]+=1
    return x

class EntityClassificationRGCN(LightningModule):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, num_relations, num_entities, l2lambda=0.01, optimizer=Adam, lr=0.01,
                 simplified=False, bias = False, **kwargs):
        super(EntityClassificationRGCN, self).__init__()
        self.num_relations = num_relations
        self.simplified = simplified
        # self.layers = ModuleList([RGCNConv(in_dim,hidden_dim,num_relations,bias=False,root_weight=True,aggr='add')]+[RGCNConv(hidden_dim,hidden_dim,num_relations,bias=False,root_weight=True,aggr='add') for _ in range(num_layers-2)]+[RGCNConv(hidden_dim,out_dim,num_relations,bias=False,root_weight=True,aggr='add')])
        if simplified:
            self.encoder = Linear(num_relations, hidden_dim)
            self.layers = ModuleList(
                [RGCNLayer(hidden_dim, hidden_dim, num_relations, num_entities, bias=bias, **kwargs) for _ in range(num_layers - 1)] + [
                    RGCNLayer(hidden_dim, out_dim, num_relations, num_entities, bias=False, activation=lambda x: x, **kwargs)])
        else:
            self.layers = ModuleList([RGCNLayer(in_dim, hidden_dim, num_relations, num_entities, bias=bias, **kwargs)] + [
                RGCNLayer(hidden_dim, hidden_dim, num_relations, num_entities, bias=bias **kwargs) for _ in range(num_layers - 2)] + [
                                         RGCNLayer(hidden_dim, out_dim, num_relations, num_entities, bias=False, activation=lambda x: x,
                                                   **kwargs)])
        self.loss = CrossEntropyLoss(reduction='sum')
        self.optimizer = optimizer
        self.lr = lr
        self.l2lambda = l2lambda
        self.save_hyperparameters()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        self.fin_accuracy = None

    def forward(self, x, edge_index, edge_types, edge_attributes, num_nodes):
        x = self.encode(x, edge_index, edge_attributes, edge_types, num_nodes)
        return softmax(x, -1)

    def encode(self, x, edge_index, edge_attributes, edge_types, num_nodes):
        if self.simplified:
            x = zeros((num_nodes, self.num_relations),dtype=long, device = edge_index.device).float()
            x = generate_feat_features(x, edge_index, edge_types)
            x = self.encoder(x)
        if x is None:  # If there are no initial features, just use a one to encode it
            pass
            # x = one_hot(as_tensor([i for i in range(num_nodes)], dtype=long, device = edge_index.device)).float()
        for l in self.layers:
            x = l(x, edge_index, edge_attributes)
        return x

    def training_step(self, batch, batch_idx):
        x, edge_index, edge_types, y = batch.x, batch.edge_index, batch.edge_type, batch.train_y[:int(0.8*len(batch.train_y))]
        edge_attributes = one_hot(edge_types, num_classes=self.num_relations)
        x = self.encode(x, edge_index, edge_attributes, edge_types, batch.num_nodes)
        # CODE UP TO HERE IS KINDA NASTY -- SHOULD WORK ON MAKING DATALOADERS MORE STANDARDISED...
        x = softmax(x[batch.train_idx[:int(0.8*len(batch.train_y))]], -1)
        loss = self.loss(x, y)
        for name, param in self.layers[0].named_parameters():  # L2 Loss
            loss += norm(param) * self.l2lambda
        self.log("train_loss", loss.item())
        y_pred = max(x, -1).indices
        self.train_accuracy(y_pred, y)
        self.log("train_acc", self.train_accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_index, edge_types, y = batch.x, batch.edge_index, batch.edge_type, batch.train_y[int(0.8*len(batch.train_y)):]
        edge_attributes = one_hot(edge_types, num_classes=self.num_relations)
        x = self.encode(x, edge_index, edge_attributes, edge_types, batch.num_nodes)
        # CODE UP TO HERE IS KINDA NASTY -- SHOULD WORK ON MAKING DATALOADERS MORE STANDARDISED...
        x = softmax(x[batch.train_idx[int(0.8*len(batch.train_y)):]], -1)
        loss = self.loss(x, y)
        for name, param in self.layers[0].named_parameters():  # L2 Loss
            loss += norm(param) * self.l2lambda
        self.log("validation_loss", loss.item())
        y_pred = max(x, -1).indices
        self.val_accuracy(y_pred, y)
        self.log("validation_acc", self.val_accuracy)
        self.fin_accuracy = torchmetrics.functional.accuracy(y_pred,y).item()
        return loss

    def train_step_end(self, outs):
        self.log("train_epoch_acc", self.train_accuracy)

    def test_step(self, batch, batch_idx):
        x, edge_index, edge_types, y = batch.x, batch.edge_index, batch.edge_type, batch.test_y
        edge_attributes = one_hot(edge_types, num_classes=self.num_relations)
        x = self.encode(x, edge_index, edge_attributes, edge_types, batch.num_nodes)
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
