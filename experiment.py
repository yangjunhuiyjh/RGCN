from pytorch_lightning import Trainer
from Models.LPLightningModel import LinkPredictionDistMult, LinkPredictionRGCN
from train import train_ec, train_lp
from DataLoaders.dataloader import get_dataset
from torch_geometric.loader import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
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
    parser.add_argument('--ensemble_alpha', type=float, default=1.0)
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
    parser.add_argument('--hyperparameter_tuning', default=False, action='store_true',
                        help='hyperparameter tuning')
    parser.add_argument('--bias', default=False, action='store_true', help='use bias?')
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
        logger = TensorBoardLogger('Logs/')

    if args.ensemble:
        model = LinkPredictionRGCN.load_from_checkpoint("trained_models/lp_wn18.ckpt")
        model.setup_test
        trainer = Trainer()
        trainer.test(model,dl) ## comment out if results already present
        distmult = LinkPredictionDistMult.load_from_checkpoint("trained_models/lp_wn18_distmult.ckpt")
        trainer.test(distmult,dl) ## comment out if results already present
        model.make_ensemble(distmult, args.ensemble_alpha)
        trainer.test(model, dl)
    elif args.task == 'ec':
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
                                      out_dim=args.out_dim, num_gpus=args.num_gpus,
                                      callbacks=[PyTorchLightningPruningCallback(trial, monitor='validation_acc'),
                                                 EarlyStopping(monitor="validation_loss", mode="min")], bias = args.bias,
                                      **trial_params)
            res = model.fin_accuracy
            return res

        if args.hyperparameter_tuning:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=10)
            new_params = study.best_params
            print("the best parameters are", new_params)
        else:
            new_params = {'num_bases': args.num_bases, 'l2param': args.l2param, 'lr': args.lr}

        results = []
        for _ in range(10):
            model, trainer = train_ec(logger, dl, args.num_epoch, num_nodes, num_relation_types,
                                      norm_type=args.norm_type, hidden_dim=args.hidden_dim,
                                      out_dim=args.out_dim, num_gpus=args.num_gpus,
                                      callbacks=[],
                                      **new_params)
            results.append(trainer.test(model, dl))
        acc = 0
        print(results)

        for result in results:
            print(result)
            acc += result[0]['test_epoch_acc']
        acc /= len(results)
        print("params", new_params)
        print("Average accuracy for the best parameters:", acc)
    elif args.task == 'lp':
        validation_params = {
            'hidden_dim': [100, 200, 400],
             # 'num_blocks': [None, 20, 50],
            'lr': [1e-2, 1e-3],
            'l2param': [0, 5e-4]
        }


        def objective(trial):
            trial_params = {}
            for param_name, values in validation_params.items():
                trial_params[param_name] = trial.suggest_categorical(param_name, values)
            model, trainer = train_lp(logger, dl, 1, 50, num_nodes, num_relation_types,
                                      norm_type=args.norm_type,
                                      num_gpus=args.num_gpus,
                                      callbacks=[
                                          PyTorchLightningPruningCallback(trial, monitor='validation_loss'),
                                          EarlyStopping(monitor="validation_loss", mode="min")],
                                      model=args.model, bias=args.bias,
                                      **trial_params)
            res = model.final_loss
            return res

        if args.hyperparameter_tuning:
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=10)
            new_params = study.best_params
            print("the best parameters are", new_params)
        else:
            new_params = {'num_bases': args.num_bases, 'hidden_dim': args.hidden_dim, 'lr': args.lr, 'l2param': args.l2param}

        results = []
        for _ in range(1):
            model, trainer = train_lp(logger, dl, 1, args.num_epoch, num_nodes, num_relation_types,
                                      norm_type=args.norm_type,
                                      num_gpus=args.num_gpus, model=args.model, bias = args.bias,
                                      **new_params)
            # if args.ensemble:
            #     distmult, _ = train_distmult(...)
            #     model.make_ensemble(distmult)
            if args.model == "rgcn":
                model.setup_test()
            print("saving model")
            trainer.save_checkpoint(args.model_output_path)
            results.append(trainer.test(model, dl))
        print(results)

        raw_mrr = 0
        filtered_mrr = 0
        hits1 = 0
        hits3 = 0
        hits10 = 0

        for result in results:
            raw_mrr += result[0]['test_raw_mrr']
            filtered_mrr += result[0]["test_filtered_mrr"]
            hits1 += result[0]["test_hits@1"]
            hits3 += result[0]["test_hits@3"]
            hits10 += result[0]["test_hits@10"]
        len_samples = len(results)
        print(raw_mrr / len_samples, filtered_mrr / len_samples, hits1 / len_samples, hits3 / len_samples,
              hits10 / len_samples)
    else:
        raise ValueError('invalid task!')
