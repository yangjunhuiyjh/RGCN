from train import train_ec, train_lp
from DataLoaders.dataloader import get_dataset
from torch_geometric.loader import DataLoader
from pytorch_lightning.loggers import NeptuneLogger
import argparse
import optuna
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='aifb')
    parser.add_argument('--model', type=str, default='rgcn')
    parser.add_argument('--task', type=str, default='ec')
    parser.add_argument('--num_epoch', type=int, default=100, help='number of maximum epochs')
    parser.add_argument('--num_edge_types', type=int, default=90)
    parser.add_argument('--norm_type', type=str, default='relation-degree',
                        help='normalisation method')
    parser.add_argument('--l2param', type=float, default=0.01)
    parser.add_argument('--num_bases', type=int, default=30)
    parser.add_argument('--num_blocks', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--out_dim', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-2, help='the learning rate used by the optimizer')
    parser.add_argument('--model_output_path', type=str, default='trained_models/model.ckpt',
                        help='Where model is stored after training')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='use small training set and disable logging for quickfire testing')
    parser.add_argument('--num_gpus', type=int, default=0, help='number of gpus to be used')
    parser.add_argument('--ensemble', default=False, action='store_true',
                        help='use the ensemble between rgcn and distmult for link prediction')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    ds = get_dataset(args.dataset)
    dl = DataLoader(ds)
    num_nodes = ds[0].num_nodes
    num_relation_types = args.num_edge_types

    if args.debug:
        logger = None
    else:
        logger = NeptuneLogger(project='dylanslavinhillier/ATML',
                               api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZTZiNDUxYi03ZDNiLTQ3N2EtYjQwMC0wZjA0NTJiNTgwZDQifQ==")

    if args.task == 'ec':
        validation_params = {
            'l2param': [0, 5e-4],
            'num_bases': [None, 10, 20, 40, 80],
            'lr': [1e-2, 1e-3, 1e-4]
        }

        def objective(trial):
            trial_params = {}
            for param_name, values in validation_params.items():
                trial_params[param_name] = trial.suggest_categorical(param_name, values)
            model, trainer = train_ec(logger, dl, 30, num_nodes, num_relation_types,
                                      norm_type=args.norm_type, hidden_dim=args.hidden_dim,
                                      out_dim=args.out_dim, num_gpus=args.num_gpus, callbacks=[PyTorchLightningPruningCallback(trial, monitor='validation_acc'), EarlyStopping(monitor="validation_loss", mode="min")],
                                      **trial_params)
            res = model.fin_accuracy
            return res

        study = optuna.create_study(direction='maximize')
        study.optimize(objective,n_trials=10)
        new_params = study.best_params
        print("the best parameters are", new_params)
        results = []
        for _ in range(10):
            model, trainer = train_ec(logger, dl, args.num_epoch, num_nodes, num_relation_types,
                                      norm_type=args.norm_type, hidden_dim=args.hidden_dim,
                                      out_dim=args.out_dim, num_gpus=args.num_gpus, callbacks=[EarlyStopping(monitor="validation_loss", mode="min")],
                                      **new_params)
            results.append(trainer.test(model, dl))
        acc = 0
        print(results)

        for result in results:
            print(result)
            acc += result[0]['test_epoch_acc']
        acc /= len(results)
        print("Average accuracy for the best parameters:", acc)
    elif args.task == 'lp':
        pass
        # for _ in range(10):
        #     model, trainer = train_lp(logger, dl, args.num_epoch, num_nodes, num_relation_types, norm_type=args.norm_type,
        #                               num_blocks=args.num_blocks, hidden_dim=args.hidden_dim,
        #                               lr=args.lr, num_gpus=args.num_gpus, model=args.model)
        #     if args.ensemble:
        #         distmult, _ = train_distmult(...)
        #         model.make_ensemble(distmult)

        #     results.append(trainer.test(model, dl))
    else:
        raise ValueError('invalid task!')
