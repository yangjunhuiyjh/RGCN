from Models.LPLightningModel import LinkPredictionRGCN
from DataLoaders.dataloader import get_dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

model = LinkPredictionRGCN.load_from_checkpoint("trained_models/model.ckpt")
dataset = get_dataset("fb15k237")
dl = DataLoader(dataset)


def is_valid_triple(s, r, o, graph):
    edge_index = graph.test_edge_index
    edge_types = graph.test_edge_type
    for e in range(len(edge_index)):
        if not edge_types[e] == r:
            continue
        if not edge_index[e][0] == s:
            continue
        if not edge_index[e][1] == o:
            continue
        return True
    return False


def calc_mrr(ranks):
    res = 0
    for rank in ranks:
        res += 1 / rank
    return res / len(ranks)


def calc_hits(ranks):
    n = len(ranks)
    hits_1 = 0
    hits_3 = 0
    hits_10 = 0

    for rank in ranks:
        if rank == 1:
            hits_1 += 1
        if rank <= 3:
            hits_3 += 1
        if rank <= 10:
            hits_10 += 1

    return [hits_1 / n, hits_3 / n, hits_10 / n]


for graph in dl:
    ranks = []
    filtered_ranks = []
    edge_index = graph.test_edge_index
    edge_types = graph.test_edge_type
    x = model.forward(edge_index, edge_types)

    for edge in tqdm(range(graph.test_edge_index.size(1))):
        edge_idxs = edge_index[:, edge]
        edge_score = model.score(edge_idxs[0], edge_types[edge], edge_idxs[1], x).squeeze().item()
        rank_s, filtered_rank_s = 1, 1
        rank_o, filtered_rank_o = 1, 1

        for entity in range(graph.num_nodes):
            s_score = model.score(entity, edge_types[edge], edge_idxs[1], x).squeeze().item()
            if s_score > edge_score:
                rank_s += 1
                if not is_valid_triple(entity, edge, edge_idxs[1], graph):
                    filtered_rank_s += 1
            o_score = model.score(edge_idxs[0], edge_types[edge], entity, x).squeeze().item()
            if o_score > edge_score:
                rank_o += 1
                if not is_valid_triple(edge_idxs[0], edge, entity, graph):
                    filtered_rank_o += 1

        ranks.append(rank_o)
        ranks.append(rank_s)
        filtered_ranks.append(filtered_rank_o)
        filtered_ranks.append(filtered_rank_s)

    print(ranks)
    print("Raw MRR:", calc_mrr(ranks))
    print("Filtered MRR:", calc_mrr(filtered_ranks))
    print("Hits@1, Hits@3, Hits@10:", calc_hits(filtered_ranks))
