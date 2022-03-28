from multiprocessing.sharedctypes import Value
from pytorch_lightning import Trainer
from torch_geometric.loader import DataLoader
from DataLoaders.dataloader import get_dataset
import argparse

from Models.ECLightningModel import EntityClassificationRGCN
from Models.LPLightningModel import LinkPredictionRGCN

from pytorch_lightning.loggers import NeptuneLogger

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='aifb')
    parser.add_argument('--model', type=str, default='rgcn')
    parser.add_argument('--task', type=str, default='ec')
    parser.add_argument('--num_epoch', type=int, default=100, help='number of maximum epochs')
    parser.add_argument('--num_edge_types', type=int,default=90)
    parser.add_argument('--batch_size', type=int, default=1, help='batchsize')
    parser.add_argument('--lr', type=float, default=1e-2, help='the learning rate used by the optimizer')
    parser.add_argument('--model_output_path', type=str, default='trained_models/model.ckpt',help='Where model is stored after training')
    parser.add_argument('--debug', default=False,action='store_true', help='use small training set and disable logging for quickfire testing')
    parser.add_argument('--num_gpus', type=int, default=0,help='number of gpus to be used')
    return parser.parse_args()

def train_ec(logger, dl, epochs, num_nodes, num_relation_types, l2param =0.01, norm_type='relation-degree', num_bases=30, hidden_dim=16, out_dim=4,lr=0.01,num_gpus=0):
    if num_gpus>0:
        trainer = Trainer(logger=logger,log_every_n_steps=1,max_epochs=epochs,gpus=num_gpus,enable_checkpointing=False,strategy='ddp')
    else:
        trainer = Trainer(logger=logger,log_every_n_steps=1,max_epochs=epochs,gpus=num_gpus,enable_checkpointing=False)
    model = EntityClassificationRGCN(2,num_nodes,hidden_dim,out_dim,num_relation_types,num_bases=num_bases,l2lambda=l2param,lr=lr, norm_type=norm_type)
    trainer.fit(model,dl)
    return model, trainer

def train_lp(logger, dl, epochs, num_nodes, num_relations, norm_type='non-relation-degree', num_blocks=100, hidden_dim=500, lr =0.01, num_gpus=0):
    if num_gpus>0:
        trainer = Trainer(logger=logger,log_every_n_steps=1,max_epochs=epochs,gpus=num_gpus,enable_checkpointing=False,strategy='ddp')
    else:
        trainer = Trainer(logger=logger,log_every_n_steps=1,max_epochs=epochs,gpus=num_gpus,enable_checkpointing=False)
    model = LinkPredictionRGCN(2,hidden_dim,hidden_dim,hidden_dim,num_relations,num_nodes,num_blocks=num_blocks,norm_type=norm_type, lr=lr)   
    trainer.fit(model,dl)
    return model, trainer

if __name__ == '__main__':
    args = parse_arguments()
    ds = get_dataset(args.dataset)
    print(ds[0])
    num_nodes = ds[0].num_nodes
    num_relations = ds[0].edge_type.size(0)
    num_relation_types = args.num_edge_types
    print(num_nodes,num_relations)
    # raise ValueError("oops")

    if args.debug:
        logger = None
    else:
        logger= NeptuneLogger(project='dylanslavinhillier/ATML',api_key="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZTZiNDUxYi03ZDNiLTQ3N2EtYjQwMC0wZjA0NTJiNTgwZDQifQ==")
        
 
    ds = get_dataset(args.dataset)
    print(len(ds),ds,type(ds))
    dl = DataLoader(ds,args.batch_size)
    if args.task == 'ec':
        model, trainer = train_ec(logger,dl,args.num_epoch,num_nodes,num_relation_types)
    elif args.task == 'lp':
        model, trainer = train_lp(logger,dl,args.num_epoch,num_nodes,num_relations)
    print(model)
    trainer.fit(model,dl)
    trainer.save_checkpoint(args.model_output_path)