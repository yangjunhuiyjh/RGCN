from Models.LPLightningModel import LinkPredictionRGCN
from DataLoaders.dataloader import get_dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

model = LinkPredictionRGCN.load_from_checkpoint("trained_models/model.ckpt")
dataset = get_dataset("fb15k237")
dl = DataLoader(dataset)

def get_existent_edges(p):
    pass

for graph in dl:
    ranks = []
    edge_index=graph.test_edge_index
    edge_types=graph.test_edge_type
    x = model.forward(edge_index,edge_types)
    for edge in tqdm(range(graph.test_edge_index.size(1))):
        edge_idxs = edge_index[:,edge]
        edge_score = model.score(edge_idxs[0],edge_types[edge],edge_idxs[1],x).squeeze().item()
        rank_s = 1
        rank_o = 1
        for entity in range(graph.num_nodes):
            s_score = model.score(entity, edge_types[edge],edge_idxs[1],x).squeeze().item()
            if s_score > edge_score:
                rank_s+=1
            o_score = model.score(edge_idxs[0],edge_types[edge],entity,x).squeeze().item()
            if o_score > edge_score:
                rank_o+=1
        ranks.append(rank_o)
        ranks.append(rank_s)
print(ranks)