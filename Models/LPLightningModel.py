from sklearn import ensemble
from torch import as_tensor, norm, long, full, sigmoid
from torch.nn.functional import one_hot
from pytorch_lightning import LightningModule
from torch.nn import ModuleList, BCELoss, Linear
from torch.optim import Adam
from Models.rgcn import RGCNLayer
from Models.DistMult import distMult
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
from util import test_graph


class LinkPredictionRGCN(LightningModule):
    def __init__(self, num_layers, hidden_dim, num_relation_types, num_entities, omega=1,
                 l2lambda=0.01, optimizer=Adam, lr=0.01, ensemble_alpha=1, dropout_ratio=0.2, **kwargs):
        '''
        ensemble_alpha: ensemble weight parameter on LPRGCN (alpha*rgcn+(1-alpha)*distmult)
        dropout_ratio: edge dropout parameter
        **kwargs gets forwarded to rgcn
        omega: negative sample rate

        To run testing call:
        model.setup_test()
        x = model.forward(edge_index,edge_types)
        score = model.score(s,p,o,x) ## s: subject, p: relation, o: object
        '''
        super(LinkPredictionRGCN, self).__init__()
        self.num_relation_types = num_relation_types
        self.num_entities = num_entities
        self.layers = ModuleList(
            [RGCNLayer(hidden_dim, hidden_dim, num_relation_types, dropout=dropout_ratio, **kwargs) for _ in
             range(num_layers)])
        self.embedder = Linear(num_entities, hidden_dim)
        self.loss = BCELoss(reduction='sum')
        self.distmult = distMult(hidden_dim, num_relation_types)
        self.ensemble_alpha = ensemble_alpha
        self.ensemble_distmult = None
        self.optimizer = optimizer
        self.omega = omega
        self.num_entities = num_entities
        self.lr = lr
        self.l2lambda = l2lambda
        self.save_hyperparameters()
        self.final_loss = None

    def make_ensemble(self, distmult):
        self.ensemble_distmult = distmult

    def setup_test(self):
        for layer in self.layers:
            layer.dropout = None

    def forward(self, edge_index, edge_types):
        '''
        edge_index: [2,num_edges]
        edge_types: [num_edges]
        '''
        x = one_hot(as_tensor([i for i in range(self.num_entities)], dtype=long, device=edge_index.device)).float()
        x = self.embedder(x)
        edge_attributes = one_hot(edge_types, num_classes=self.num_relation_types)
        for layer in self.layers:
            x = layer(x, edge_index, edge_attributes)
        return x

    def batch_edges(self, edges, edge_type, label, batch_size=10000):
        batched_edges = []
        for i in range(0, edges.size(1), batch_size):
            batch = (
                edges[:, i:i + batch_size], edge_type[i:i + batch_size],
                full((min(batch_size, edges.size(1) - i),), label, device=edges.device))
            batched_edges.append(batch)
        return batched_edges

    def training_step(self, batch):
        '''
        batch should contain train_edge_index [2,num_edges], train_edge_type [num_edges]
        '''
        edge_index = batch.train_edge_index
        edge_attributes = batch.train_edge_type
        edge_attributes = one_hot(edge_attributes, num_classes=self.num_relation_types)
        x = one_hot(as_tensor([i for i in range(self.num_entities)], dtype=long, device=edge_index.device)).float()
        x = self.embedder(x)
        for l in self.layers:
            x = l(x, edge_index, edge_attributes)
        loss = 0
        for _ in range(self.omega):
            edges = negative_sampling(edge_index, self.num_entities)
            batches = self.batch_edges(edges, batch.train_edge_type, 0.0)
            for (edge, edge_attribute, label) in tqdm(batches):
                score = sigmoid(self.distmult(x[edge[0]], edge_attribute, x[edge[1]]))
                loss += self.loss(score, label) / (edge_attribute.size(0) * (1 + self.omega))
        edges = edge_index
        batches = self.batch_edges(edges, batch.train_edge_type, 1.0)
        for (edge, edge_attribute, label) in tqdm(batches):
            score = sigmoid(self.distmult(x[edge[0]], edge_attribute, x[edge[1]]))
            loss += self.loss(score, label) / (edge_attribute.size(0) * (1 + self.omega))
        for name, param in self.distmult.named_parameters():  ## L2 Loss
            loss += norm(param) * self.l2lambda
        self.log("train_loss", loss.item())
        return loss

    def test_step(self, batch, batch_idx):
        results = test_graph(self, self.num_entities, batch.train_edge_index, batch.train_edge_type,
                             batch.test_edge_index, batch.test_edge_type)
        for key, value in results.items():
            self.log("test_" + key, value)

    def validation_step(self, batch, batch_idx):
        edge_index = batch.valid_edge_index
        edge_attributes = batch.valid_edge_type
        edge_attributes = one_hot(edge_attributes, num_classes=self.num_relation_types)
        x = one_hot(as_tensor([i for i in range(self.num_entities)], dtype=long, device=edge_index.device)).float()
        x = self.embedder(x)
        for l in self.layers:
            x = l(x, edge_index, edge_attributes)
        loss = 0
        for _ in range(self.omega):
            edges = negative_sampling(edge_index, self.num_entities)
            batches = self.batch_edges(edges, batch.valid_edge_type, 0.0)
            for (edge, edge_attribute, label) in tqdm(batches):
                score = sigmoid(self.distmult(x[edge[0]], edge_attribute, x[edge[1]]))
                loss += self.loss(score, label) / (edge_attribute.size(0) * (1 + self.omega))
        edges = edge_index
        batches = self.batch_edges(edges, batch.valid_edge_type, 1.0)
        for (edge, edge_attribute, label) in tqdm(batches):
            score = sigmoid(self.distmult(x[edge[0]], edge_attribute, x[edge[1]]))
            loss += self.loss(score, label) / (edge_attribute.size(0) * (1 + self.omega))
        for name, param in self.distmult.named_parameters():  ## L2 Loss
            loss += norm(param) * self.l2lambda
        self.log("validation_loss", loss.item())
        self.final_loss = loss.item()
        return loss
        # results = test_graph(self, self.num_entities, batch.train_edge_index, batch.train_edge_type,
        #           batch.valid_edge_index, batch.valid_edge_type)
        # for key,value in results:
        #     self.log("validation_"+key,value)

    def score(self, s, p, o, x):
        """
        s: [batchsize,index] of subject
        p: [batchsize,index] of relation
        o: [batchsize,index] of object
        """
        score = self.distmult(x[s], p, x[o])
        if self.ensemble_distmult:
            x_distmult = self.ensemble_distmult.forward()
            distmult_score = self.ensemble_distmult(x_distmult[s], p, x_distmult[o])
            score = self.ensemble_alpha * score + (1 - self.ensemble_alpha) * distmult_score
        return score

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)


class LinkPredictionDistMult(LightningModule):
    def __init__(self, hidden_dim, num_relation_types, num_entities, omega=1,
                 l2lambda=0.01, optimizer=Adam, lr=0.01, **kwargs):
        '''
        omega: negative sample rate

        To run testing call:
        x = model.forward(edge_index,edge_types)
        score = model.score(s,p,o,x) ## s: subject, p: relation, o: object
        '''
        super(LinkPredictionDistMult, self).__init__()
        self.num_relation_types = num_relation_types
        self.num_entities = num_entities
        self.embedder = Linear(num_entities, hidden_dim)
        self.loss = BCELoss(reduction='sum')
        self.distmult = distMult(hidden_dim, num_relation_types)
        self.optimizer = optimizer
        self.omega = omega
        self.num_entities = num_entities
        self.lr = lr
        self.l2lambda = l2lambda
        self.save_hyperparameters()
        self.final_loss = None

    def forward(self, edge_index, edge_types):
        x = one_hot(as_tensor([i for i in range(self.num_entities)], dtype=long, device=edge_index.device)).float()
        x = self.embedder(x)
        return x

    def batch_edges(self, edges, edge_type, label, batch_size=1000):
        batched_edges = []
        for i in range(0, edges.size(1), batch_size):
            batch = (
                edges[:, i:i + batch_size], edge_type[i:i + batch_size],
                full((min(batch_size, edges.size(1) - i),), label))
            batched_edges.append(batch)
        return batched_edges

    def training_step(self, batch):
        """
        batch should contain train_edge_index [2,num_edges], train_edge_type [num_edges]
        """
        edge_index = batch.train_edge_index
        edge_attributes = batch.train_edge_type
        edge_attributes = one_hot(edge_attributes, num_classes=self.num_relation_types, device=edge_index.device)
        x = one_hot(as_tensor([i for i in range(self.num_entities)], dtype=long, device=edge_index.device)).float()
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

    def test_step(self, batch, batch_idx):
        results = test_graph(self, self.num_entities, batch.train_edge_index, batch.train_edge_type,
                             batch.test_edge_index, batch.test_edge_type)
        for key, value in results.items():
            self.log("test_" + key, value)

    def validation_step(self, batch, batch_idx):
        edge_index = batch.valid_edge_index
        edge_attributes = batch.valid_edge_type
        edge_attributes = one_hot(edge_attributes, num_classes=self.num_relation_types, device=edge_index.device)
        x = one_hot(as_tensor([i for i in range(self.num_entities)], dtype=long, device=edge_index.device)).float()
        x = self.embedder(x)
        for l in self.layers:
            x = l(x, edge_index, edge_attributes)
        loss = 0
        for _ in range(self.omega):
            edges = negative_sampling(edge_index, self.num_entities)
            batches = self.batch_edges(edges, batch.valid_edge_type, 0.0)
            for (edge, edge_attribute, label) in tqdm(batches):
                score = sigmoid(self.distmult(x[edge[0]], edge_attribute, x[edge[1]]))
                loss += self.loss(score, label) / (edge_attribute.size(0) * (1 + self.omega))
        edges = edge_index
        batches = self.batch_edges(edges, batch.valid_edge_type, 1.0)
        for (edge, edge_attribute, label) in tqdm(batches):
            score = sigmoid(self.distmult(x[edge[0]], edge_attribute, x[edge[1]]))
            loss += self.loss(score, label) / (edge_attribute.size(0) * (1 + self.omega))
        for name, param in self.distmult.named_parameters():  ## L2 Loss
            loss += norm(param) * self.l2lambda
        self.log("validation_loss", loss.item())
        self.final_loss = loss.item()
        return loss
        # results = test_graph(self, self.num_entities, batch.train_edge_index, batch.train_edge_type,
        #           batch.valid_edge_index, batch.valid_edge_type)
        # for key,value in results:
        #     self.log("validation_"+key,value)

    def score(self, s, p, o, x):
        """
        s: [batchsize,index] of subject
        p: [batchsize,index] of relation
        o: [batchsize,index] of object
        """
        score = self.distmult(x[s], p, x[o])
        return score

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)
