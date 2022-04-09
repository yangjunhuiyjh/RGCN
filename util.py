import enum
from torch import full, long, ones_like, LongTensor, tensor
from tqdm import tqdm
from torch_geometric import utils


def generate_invalid_masks_subj(subjects, relation, object, edge_index, edge_types):
    valid_edges = edge_index.T[(edge_types.eq(relation) * edge_index[1, :].eq(object))].T ## outputs the valid edges
    valid_subjects = valid_edges[0]
    mask = ones_like(subjects, device=edge_index.device)
    mask[valid_subjects] = 0
    return mask


def generate_invalid_masks_obj(subject, relation, objects, edge_index, edge_types):
    valid_edges = edge_index.T[(edge_types.eq(relation) * edge_index[0, :].eq(subject))].T
    valid_objects = valid_edges[1]
    mask = ones_like(objects, device=edge_index.device)
    mask[valid_objects] = 0
    return mask


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


def test_graph(model, num_entities, train_edge_index, train_edge_types, test_edge_index, test_edge_types, all_edge_index, all_edge_types, by_degree=False):
    ranks = []
    filtered_ranks = []
    avg_degree = []
    x = model.forward(train_edge_index, train_edge_types)
    node_degree = utils.degree(all_edge_index,num_entities,long)
    for edge in tqdm(range(test_edge_index.size(1))):
        test_edge = test_edge_index[:, edge]
        edge_score = model.score(test_edge[0], test_edge_types[edge], test_edge[1], x).squeeze().item()
        rank_s, filtered_rank_s = 1, 1
        rank_o, filtered_rank_o = 1, 1
        s_score = model.score([i for i in range(num_entities)], full((num_entities,), test_edge_types[edge].item(),device=x.device),
                              full((num_entities,), test_edge[1].item(),device=x.device), x)
        s_score_masks = s_score > edge_score
        rank_s += sum(s_score_masks)

        invalid_triple_masks = generate_invalid_masks_subj(tensor([i for i in range(num_entities)],device=x.device),
                                                           test_edge_types[edge].item(), 
                                                           test_edge[1].item(), 
                                                           all_edge_index, all_edge_types)
        filtered_s_score_masks = s_score_masks * invalid_triple_masks
        filtered_rank_s += sum(filtered_s_score_masks)  ## Sum for each valid

        o_score = model.score(full((num_entities,), test_edge[0].item(), device=x.device), full((num_entities,), test_edge_types[edge].item(), device=x.device),
                              [i for i in range(num_entities)], x)
        o_score_masks = o_score > edge_score
        rank_o += sum(o_score_masks)

        invalid_triple_masks = generate_invalid_masks_obj(test_edge[0].item(),
                                                          test_edge_types[edge].item(),
                                                          tensor([i for i in range(num_entities)], device=x.device),
                                                          all_edge_index, all_edge_types)
        filtered_o_score_masks = o_score_masks * invalid_triple_masks
        filtered_rank_o += sum(filtered_o_score_masks)  ## Sum for each valid

        ranks.append(rank_o)
        ranks.append(rank_s)
        filtered_ranks.append(filtered_rank_o)
        filtered_ranks.append(filtered_rank_s)
        avg_degree.append((node_degree[test_edge[0].item]+node_degree[test_edge[1].item])/2)
    if by_degree:
        max_deg = max(avg_degree).item()
        bins = [(lambda x: x>max_deg//10*i and x<max_deg//10*(i+1),[]) for i in range(10)]
        bin_scores = []
        for i,e in enumerate(avg_degree):
            for bin in bins:
                if bin[0](e):
                    bin[1].append(filtered_ranks[2*i])
                    bin[1].append(filtered_ranks[2*i+1])
            for bin in bins:
                mrr = calc_mrr(bin[1])
                bin_scores.append((bin[0],mrr))
        return bin_scores
    hits_1, hits_3, hits_10 = calc_hits(filtered_ranks)
    results = {
        'raw_mrr': calc_mrr(ranks),
        'filtered_mrr': calc_mrr(filtered_ranks),
        'hits@1': hits_1,
        'hits@3': hits_3,
        'hits@10': hits_10
    }

    return results