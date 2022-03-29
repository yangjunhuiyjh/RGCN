# import optuna
# from train import run_train

# ## Sample code for an experiment
# def objective(trial: optuna.trial.BaseTrial):
#     lr = trial.suggest_loguniform("lr")
#     model = run_train(lr)
#     score = run_test(model,ds)

from train import train_ec
from DataLoaders.dataloader import get_dataset
from torch_geometric.loader import DataLoader
from pytorch_lightning.loggers import NeptuneLogger
if __name__=='__main__':
    ds = get_dataset("mutag")
    dl = DataLoader(ds)
    num_nodes = ds[0].num_nodes
    results = []
    logger= NeptuneLogger(project='dylanslavinhillier/ATML',api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZTZiNDUxYi03ZDNiLTQ3N2EtYjQwMC0wZjA0NTJiNTgwZDQifQ==")

    for i in range(10):
        model, trainer = train_ec(logger,dl,50,num_nodes,46,num_bases=30,l2param=5e-4,norm_type='relation-degree',out_dim=2)
        results.append(trainer.test(model,dl))
    acc = 0
    for result in results:
        print(result)
        acc += result["test_epoch_acc"]
    acc/=len(results)
    print(acc)
        