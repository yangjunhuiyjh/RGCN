from multiprocessing.sharedctypes import Value
from pytorch_lightning import Trainer
from torch_geometric.loader import DataLoader
from DataLoaders.dataloader import get_dataset
import argparse

from Models.ECLightningModel import EntityClassificationRGCN
from Models.LPLightningModel import LinkPredictionRGCN, LinkPredictionDistMult

from pytorch_lightning.loggers import NeptuneLogger


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

<<<<<<< HEAD
def train_ec(logger, dl, epochs, num_nodes, num_relation_types, l2param =0.01, norm_type='relation-degree', num_bases=30, hidden_dim=16, out_dim=4,lr=0.01,num_gpus=0, simplified=False):
    if num_gpus>0:
        trainer = Trainer(logger=logger,log_every_n_steps=1,max_epochs=epochs,gpus=num_gpus,enable_checkpointing=False,strategy='ddp')
    else:
        trainer = Trainer(logger=logger,log_every_n_steps=1,max_epochs=epochs,gpus=num_gpus,enable_checkpointing=False)
    model = EntityClassificationRGCN(2,num_nodes,hidden_dim,out_dim,num_relation_types,num_bases=num_bases,l2lambda=l2param,lr=lr, norm_type=norm_type, simplified=simplified)
    trainer.fit(model,dl)
=======

def train_ec(logger, dl, epochs, num_nodes, num_relation_types, l2param=0.01, norm_type='relation-degree', num_bases=30,
             hidden_dim=16, out_dim=4, lr=0.01, num_gpus=0):
    if num_gpus > 0:
        trainer = Trainer(logger=logger, log_every_n_steps=1, max_epochs=epochs, gpus=num_gpus,
                          enable_checkpointing=False, strategy='ddp')
    else:
        trainer = Trainer(logger=logger, log_every_n_steps=1, max_epochs=epochs, gpus=num_gpus,
                          enable_checkpointing=False)
    model = EntityClassificationRGCN(2, num_nodes, hidden_dim, out_dim, num_relation_types, num_bases=num_bases,
                                     l2lambda=l2param, lr=lr, norm_type=norm_type)
    trainer.fit(model, dl)
>>>>>>> 83dbdbc8f0812d62f260fc0ecb30754b4589acd7
    return model, trainer


def train_lp(logger, dl, epochs, num_nodes, num_relations, norm_type='non-relation-degree', num_blocks=100,
             hidden_dim=500, lr=0.01, num_gpus=0, model='rgcn'):
    if num_gpus > 0:
        trainer = Trainer(logger=logger, log_every_n_steps=1, max_epochs=epochs, gpus=num_gpus,
                          enable_checkpointing=False, strategy='ddp')
    else:
        trainer = Trainer(logger=logger, log_every_n_steps=1, max_epochs=epochs, gpus=num_gpus,
                          enable_checkpointing=False)
    if model == 'rgcn':
        model = LinkPredictionRGCN(2, hidden_dim, hidden_dim, hidden_dim, num_relations, num_nodes,
                                   num_blocks=num_blocks, norm_type=norm_type, lr=lr)
    elif model == 'distmult':
        model = LinkPredictionDistMult(hidden_dim, num_relations, num_nodes,
                                       num_blocks=num_blocks, norm_type=norm_type, lr=lr)
    trainer.fit(model, dl)
    return model, trainer


if __name__ == '__main__':
    args = parse_arguments()
    ds = get_dataset(args.dataset)
    print(ds[0])
    num_nodes = ds[0].num_nodes
    num_relations = ds[0].edge_type.size(0)
    num_relation_types = args.num_edge_types
    print(num_nodes, num_relations)

    if args.debug:
        logger = None
    else:
        logger = NeptuneLogger(project='dylanslavinhillier/ATML',
                               api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZTZiNDUxYi03ZDNiLTQ3N2EtYjQwMC0wZjA0NTJiNTgwZDQifQ==")

    ds = get_dataset(args.dataset)
    print(len(ds), ds, type(ds))
    dl = DataLoader(ds, args.batch_size)
    if args.task == 'ec':
        model, trainer = train_ec(logger, dl, args.num_epoch, num_nodes, num_relation_types)
    elif args.task == 'lp':
        model, trainer = train_lp(logger, dl, args.num_epoch, num_nodes, num_relations, args.model)
    else:
        raise ValueError("invalid task!")

    print(model)
    trainer.fit(model, dl)
    trainer.save_checkpoint(args.model_output_path)
