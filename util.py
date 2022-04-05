from numpy import empty_like, ones, ones_like
from torch import full, stack
from tqdm import tqdm


def generate_invalid_masks_subj(subjects, relation, object, edge_index, edge_types):
    valid_edge_mask = edge_index.T.where(edge_types == relation and edge_index[1] == object).T
    valid_subjects = edge_index[0, valid_edge_mask]
    mask = ones_like(subjects)
    mask[valid_subjects]= False
    return mask

def generate_invalid_masks_obj(subject, relation, objects, edge_index, edge_types):
    valid_edge_mask = edge_index.T.where(edge_types == relation and edge_index[0] == subject).T
    valid_objects = edge_index[1, valid_edge_mask]
    mask = ones_like(objects)
    mask[valid_objects]= False
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


def test_graph(model, num_entities, train_edge_index, train_edge_types, test_edge_index, test_edge_types):
    ranks = []
    filtered_ranks = []
    x = model.forward(train_edge_index, train_edge_types)
    for edge in tqdm(range(test_edge_index.size(1))):
        test_edge = test_edge_index[:, edge]
        edge_score = model.score(test_edge[0], test_edge_types[edge], test_edge[1], x).squeeze().item()
        rank_s, filtered_rank_s = 1, 1
        rank_o, filtered_rank_o = 1, 1
        s_score = model.score([i for i in range(num_entities)], full(num_entities,test_edge_types[edge]), full(num_entities, test_edge[1]), x)
        s_score_masks = s_score > edge_score
        rank_s += sum(s_score_masks)

        invalid_triple_masks = generate_invalid_masks_subj([i for i in range(num_entities)], full(num_entities,test_edge_types[edge]), full(num_entities, test_edge[1]), test_edge, test_edge_types)
        filtered_s_score_masks = s_score_masks * invalid_triple_masks
        filtered_rank_s += sum(filtered_s_score_masks) ## Sum for each valid

        o_score = model.score(full(num_entities, test_edge[0]), full(num_entities,test_edge_types[edge]), [i for i in range(num_entities)], x)
        o_score_masks = o_score > edge_score
        rank_o += sum(o_score_masks)

        invalid_triple_masks = generate_invalid_masks_obj(full(num_entities, test_edge[0]), full(num_entities,test_edge_types[edge]), [i for i in range(num_entities)], test_edge, test_edge_types)
        filtered_o_score_masks = o_score_masks * invalid_triple_masks
        filtered_rank_o += sum(filtered_o_score_masks) ## Sum for each valid

        ranks.append(rank_o)
        ranks.append(rank_s)
        filtered_ranks.append(filtered_rank_o)
        filtered_ranks.append(filtered_rank_s)

    hits_1, hits_3, hits_10 = calc_hits(filtered_ranks)
    results = {
        'raw_mrr': calc_mrr(ranks),
        'filtered_mrr': calc_mrr(filtered_ranks),
        'hits@1': hits_1,
        'hits@3': hits_3,
        'hits@10': hits_10
    }

    return results
