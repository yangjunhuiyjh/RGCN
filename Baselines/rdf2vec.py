from torch_geometric.datasets import Entities
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG, Vertex
from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec import RDF2VecTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd


if __name__ == '__main__':
    graph = Entities("../Datasets", "AIFB")[0]

    train_entities = [str(entity.item()) for entity in graph.train_idx]
    train_labels = [str(label.item()) for label in graph.train_y]
    test_entities = [str(entity.item()) for entity in graph.test_idx]
    test_labels = [str(label.item()) for label in graph.test_y]

    entities = train_entities + test_entities
    labels = train_labels + test_labels

    print(entities)

    knowledge_graph = KG()
    for r in range(graph.edge_index.size()[-1]):
        edge = graph.edge_index[:, r]
        subj = Vertex(str(edge[0].item()))
        obj = Vertex(str(edge[1].item()))
        pred = Vertex(str(graph.edge_type[r].item()), predicate=True, vprev=subj, vnext=obj)
        knowledge_graph.add_walk(subj, pred, obj)

    transformer = RDF2VecTransformer(
        Word2Vec(epochs=10),
        walkers=[RandomWalker(4, 10, with_reverse=True, n_jobs=2)],
        verbose=1
    )

    embeddings, literals = transformer.fit_transform(knowledge_graph, entities)
    print(literals)

    train_embeddings = embeddings[: len(train_entities)]
    test_embeddings = embeddings[len(train_entities) :]

    # Fit a Support Vector Machine on train embeddings and pick the best
    # C-parameters (regularization strength).
    clf = GridSearchCV(
        SVC(), {"C": [10 ** i for i in range(-3, 4)]}
    )
    clf.fit(train_embeddings, train_labels)

    # Evaluate the Support Vector Machine on test embeddings.
    predictions = clf.predict(test_embeddings)
    print(
        f"Predicted {len(test_entities)} entities with an accuracy of "
        + f"{accuracy_score(test_labels, predictions) * 100 :.4f}%"
    )
