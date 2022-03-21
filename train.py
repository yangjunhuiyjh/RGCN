from pytorch_lightning import Trainer
from torch_geometric.loader import DataLoader
from DataLoaders.dataloader import get_dataset
import argparse

from Models.ECLightningModel import EntityClassificationRGCN


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--num_epoch', type=int, default=100, help='number of maximum epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batchsize')
    parser.add_argument('--lr', type=float, default=1e-5, help='the learning rate used by the optimizer')
    parser.add_argument('--model_output_path', type=str, default='/data/model.ckpt',help='Where model is stored after training')
    parser.add_argument('--debug', default=False,action='store_true', help='use small training set and disable logging for quickfire testing')
    parser.add_argument('--num_gpus', type=int, default=0,help='number of gpus to be used')
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_arguments()
    ds = get_dataset(args.dataset)
    if args.debug:
        logger = None
    else:
        logger= None
        
    if args.num_gpus>0:
        trainer = Trainer(logger=logger,log_every_n_steps=10,max_epochs=args.num_epoch,gpus=args.num_gpus,enable_checkpointing=False,strategy='ddp')
    else:
        trainer = Trainer(logger=logger,log_every_n_steps=10,max_epochs=args.num_epoch,gpus=args.num_gpus,enable_checkpointing=False)
    if args.model =="rgcn" and args.task =="ec":
        model = EntityClassificationRGCN(2,8285,16,4,90,num_bases=30)
    else:
        print("model not defined")
    print(model)
    ds = get_dataset(args.dataset)
    print(len(ds),ds,type(ds))
    dl = DataLoader(ds,args.batch_size)
    trainer.fit(model,dl)
