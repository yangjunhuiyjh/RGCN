from sklearn import ensemble
from torch import as_tensor, norm, long, full, sigmoid
from torch.nn.functional import one_hot
from pytorch_lightning import LightningModule
from torch.nn import ModuleList, BCELoss, Linear
from torch.optim import Adam
from Models.rgcn import RGCNLayer
from Models.DistMult import distMult
from torch_geometric.utils import negative_sampling


class LinkPredictionRGCN(LightningModule):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, num_relations, num_entities, omega=1,
                 l2lambda=0.01, optimizer=Adam, lr=0.01, ensemble_alpha=1, **kwargs):
        super(LinkPredictionRGCN, self).__init__()
        self.num_relations = num_relations
        self.num_entities = num_entities
        # self.embedder = Parameter(FloatTensor(num_entities,in_dim))
        # kaiming_normal_(self.embedder)
        self.embedder = Linear(num_entities, out_dim)
        self.layers = ModuleList([RGCNLayer(in_dim, hidden_dim, num_relations, **kwargs)] + [
            RGCNLayer(hidden_dim, hidden_dim, num_relations, **kwargs) for _ in range(num_layers - 2)] + [
                                     RGCNLayer(hidden_dim, out_dim, num_relations)])
        self.loss = BCELoss(reduction='sum')
        self.distmult = distMult(out_dim, num_relations)
        self.ensemble_alpha = ensemble_alpha
        self.ensemble_distmult = None
        self.optimizer = optimizer
        self.omega = omega
        self.num_entities = num_entities
        self.lr = lr
        self.l2lambda = l2lambda
        self.save_hyperparameters()

    def make_ensemble(self,distmult):
        self.ensemble_distmult = distmult


    def forward(self, edge_index, edge_types):
        x = one_hot(as_tensor([i for i in range(self.num_nodes)], dtype=long)).float()
        x = self.embedder(x)
        edge_attributes = one_hot(edge_types, num_classes=self.num_relations)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attributes)
        return x

    def batch_edges(self, edges, edge_type, label, batch_size=10):
        batched_edges = []
        for i in range(0, edges.size(1), batch_size):
            batch = (
                edges[:, i:i + batch_size], edge_type[i:i + batch_size],
                full((min(batch_size, edges.size(1) - i), ), label))
            batched_edges.append(batch)
        return batched_edges

    def training_step(self, batch):
        edge_index = batch.train_edge_index
        edge_attributes = batch.train_edge_type
        edge_attributes = one_hot(edge_attributes, num_classes=self.num_relations)
        x = one_hot(as_tensor([i for i in range(self.num_nodes)], dtype=long)).float()
        x = self.embedder(x)
        for l in self.layers:
            x = l(x, edge_index, edge_attributes)
        #### CODE UP TO HERE IS KINDA NASTY -- SHOULD WORK ON MAKING DATALOADERS MORE STANDARDISED...
        loss = 0
        for _ in range(self.omega):
            edges = negative_sampling(edge_index, self.num_entities)
            batches = self.batch_edges(edges, batch.train_edge_type, 0.0)
            for (edge, edge_attribute, label) in batches:
                score = sigmoid(self.distmult(x[edge[0]], edge_attribute, x[edge[1]]))
                loss += self.loss(score, label) / (edge_attribute.size(0) * (1 + self.omega))
        edges = edge_index
        batches = self.batch_edges(edges, batch.train_edge_type, 1.0)
        for (edge, edge_attribute, label) in batches:
            score = sigmoid(self.distmult(x[edge[0]], edge_attribute, x[edge[1]]))
            loss += self.loss(score, label) / (edge_attribute.size(0) * (1 + self.omega))
        for name, param in self.distmult.named_parameters():  ## L2 Loss
            loss += norm(param) * self.l2lambda
        self.log("train_loss", loss.item())
        return loss

    def test_step(self, batch):
        edge_index = batch.train_edge_index
        edge_attributes = batch.train_edge_type
        edge_attributes = one_hot(edge_attributes, num_classes=self.num_relations)
        x = one_hot(as_tensor([i for i in range(self.num_nodes)], dtype=long)).float()
        x = self.embedder(x)
        for l in self.layers:
            x = l(x, edge_index, edge_attributes)
        #### CODE UP TO HERE IS KINDA NASTY -- SHOULD WORK ON MAKING DATALOADERS MORE STANDARDISED...
        loss = 0
        for _ in range(self.omega):
            edges = negative_sampling(edge_index, self.num_entities)
            batches = self.batch_edges(edges, batch.train_edge_type, 0.0)
            for (edge, edge_attribute, label) in batches:
                score = sigmoid(self.distmult(x[edge[0]], edge_attribute, x[edge[1]]))
                if self.ensemble_distmult is not None:
                    score = self.ensemble_alpha*score + (1-self.ensemble_alpha)*sigmoid(self.ensemble_distmult(x[edge[0], edge_attribute, x[edge[1]]]))
                ### TODO measure metrics
        edges = edge_index
        batches = self.batch_edges(edges, batch.train_edge_type, 1.0)
        for (edge, edge_attribute, label) in batches:
            score = sigmoid(self.distmult(x[edge[0]], edge_attribute, x[edge[1]]))
            if self.ensemble_distmult is not None:
                score = self.ensemble_alpha*score + (1-self.ensemble_alpha)*sigmoid(self.ensemble_distmult(x[edge[0], edge_a
                ### TODO measure metrics
        for name, param in self.distmult.named_parameters():  ## L2 Loss
            loss += norm(param) * self.l2lambda
        self.log("train_loss", loss.item())
        return loss

    def score(self, s, p, o, x):
        """
        s: index of subject
        p: index of relation
        o: index of object
        """
        score = self.distmult(x[s].unsqueeze(0), p.unsqueeze(0), x[o].unsqueeze(0))
        return score

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)


class LinkPredictionDistMult(LightningModule):
    def __init__(self, out_dim, num_relations, num_entities, omega=1,
                 l2lambda=0.01, optimizer=Adam, lr=0.01, **kwargs):
        super(LinkPredictionDistMult, self).__init__()
        self.num_relations = num_relations
        self.num_entities = num_entities
        # self.embedder = Parameter(FloatTensor(num_entities,in_dim))
        # kaiming_normal_(self.embedder)
        self.embedder = Linear(num_entities, out_dim)
        self.loss = BCELoss(reduction='sum')
        self.distmult = distMult(out_dim, num_relations)
        self.optimizer = optimizer
        self.omega = omega
        self.num_entities = num_entities
        self.lr = lr
        self.l2lambda = l2lambda
        self.save_hyperparameters()

    def forward(self):
        x = one_hot(as_tensor([i for i in range(self.num_nodes)], dtype=long)).float()
        x = self.embedder(x)
        return x

    def batch_edges(self, edges, edge_type, label, batch_size=1):
        batched_edges = []
        for i in range(0, edges.size(1), batch_size):
            batch = (
                edges[:, i:i + batch_size], edge_type[i:i + batch_size],
                full((min(batch_size, edges.size(1) - i),), label))
            batched_edges.append(batch)
        return batched_edges

    def training_step(self, batch):
        edge_index = batch.train_edge_index
        edge_attributes = batch.train_edge_type
        edge_attributes = one_hot(edge_attributes, num_classes=self.num_relations)
        x = one_hot(as_tensor([i for i in range(self.num_nodes)], dtype=long)).float()
        x = self.embedder(x)
        #### CODE UP TO HERE IS KINDA NASTY -- SHOULD WORK ON MAKING DATALOADERS MORE STANDARDISED...
        loss = 0
        for _ in range(self.omega):
            edges = negative_sampling(edge_index, self.num_entities)
            batches = self.batch_edges(edges, batch.train_edge_type, 0.0)
            for (edge, edge_attribute, label) in batches:
                score = sigmoid(self.distmult(x[edge[0]], edge_attribute, x[edge[1]]))
                loss += self.loss(score, label) / (edge_attribute.size(0) * (1 + self.omega))
        edges = edge_index
        batches = self.batch_edges(edges, batch.train_edge_type, 1.0)
        for (edge, edge_attribute, label) in batches:
            score = sigmoid(self.distmult(x[edge[0]], edge_attribute, x[edge[1]]))
            loss += self.loss(score, label) / (edge_attribute.size(0) * (1 + self.omega))
        for name, param in self.distmult.named_parameters():  ## L2 Loss
            loss += norm(param) * self.l2lambda
        self.log("train_loss", loss.item())
        return loss

    def score(self, s, p, o, x):
        """
        s: index of subject
        p: index of relation
        o: index of object
        """
        score = self.distmult(x[s].unsqueeze(0), p.unsqueeze(0), x[o].unsqueeze(0))
        return score

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)
