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

if __name__=='__main__':
    ds = get_dataset("aifb")
    dl = DataLoader(ds)
    num_nodes = ds[0].num_nodes
    results = []
    for i in range(1):
        model, trainer = train_ec(None,dl,50,num_nodes,90,num_bases=None,l2param=0,norm_type='relation-degree')
        results.append(trainer.test(model,dl))
    for result in results:
        print(result)