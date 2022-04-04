from train import train_lp
from DataLoaders.dataloader import get_dataset
from torch_geometric.loader import DataLoader
from pytorch_lightning.loggers import NeptuneLogger

if __name__ == '__main__':
    ds = get_dataset("fb15k237")
    dl = DataLoader(ds)
    num_nodes = ds[0].num_nodes
    results = []
    logger = NeptuneLogger(project='dylanslavinhillier/ATML',
                           api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZTZiNDUxYi03ZDNiLTQ3N2EtYjQwMC0wZjA0NTJiNTgwZDQifQ==")

    for i in range(10):
        model, trainer = train_lp(logger, dl, 50, num_nodes, 206, norm_type='relation-degree', num_blocks=100, lr=1e-4, model='rgcn')
        results.append(trainer.test(model, dl))
    acc = 0

    for result in results:
        print(result)
        acc += result["test_epoch_acc"]
    acc /= len(results)
    print(acc)
