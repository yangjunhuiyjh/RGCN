from tqdm import tqdm


def is_valid_triple(s, r, o, edge_index, edge_types):
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


def test_graph(model, num_entities, train_edge_index, train_edge_types, test_edge_index, test_edge_types):
    ranks = []
    filtered_ranks = []
    x = model.forward(train_edge_index, train_edge_types)

    for edge in tqdm(range(test_edge_index.size(1))):
        edge_idxs = test_edge_index[:, edge]
        edge_score = model.score(edge_idxs[0], test_edge_types[edge], edge_idxs[1], x).squeeze().item()
        rank_s, filtered_rank_s = 1, 1
        rank_o, filtered_rank_o = 1, 1

        for entity in range(num_entities):
            s_score = model.score(entity, test_edge_types[edge], edge_idxs[1], x).squeeze().item()
            if s_score > edge_score:
                rank_s += 1
                if not is_valid_triple(entity, edge, edge_idxs[1], test_edge_index, test_edge_types):
                    filtered_rank_s += 1
            o_score = model.score(edge_idxs[0], test_edge_types[edge], entity, x).squeeze().item()
            if o_score > edge_score:
                rank_o += 1
                if not is_valid_triple(edge_idxs[0], edge, entity, test_edge_index, test_edge_types):
                    filtered_rank_o += 1

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
