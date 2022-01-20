#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.multiprocessing as mp

from network import Net
from dataset import MyDataset, Flip
from mainloop import train

# Training settings
parser = argparse.ArgumentParser(description='Facial Expression Classification')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--log', type=str, default="result", help='log name')
parser.add_argument('--dim1', type=int, default=32, 
                    help='Neuralnetwork dimension 1 (default: 32)')
parser.add_argument('--dim2', type=int, default=64, 
                    help='Neuralnetwork dimension 2 (default: 64)')
parser.add_argument('--dim3', type=int, default=64, 
                    help='Neuralnetwork dimension 3 (default: 64)')

args = parser.parse_args()

def main():
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    dataset_larning        = MyDataset(csvfile="train_master.tsv", flip=Flip.none, transform=True, repeat=50)
    dataset_accuracy_train = MyDataset(csvfile="train.tsv",        flip=Flip.both, transform=False, repeat=1)
    dataset_accuracy_test  = MyDataset(csvfile="test.tsv",         flip=Flip.both, transform=False, repeat=1)
    #
    kwargs = {'batch_size': args.batch_size,
                    'shuffle': True}
    if use_cuda:
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                      })

    torch.manual_seed(args.seed)

    model = Net(dim1=args.dim1, dim2=args.dim2, dim3=args.dim3).to(device)
    model.share_memory() # gradients are allocated lazily, so they are not shared here

    print("Num process:", args.num_processes)

    # try:
    #     mp.set_start_method('spawn')
    # except RuntimeError:
    #     pass
    #
    if mp.get_start_method() == 'fork':
        mp.set_start_method('spawn', force=True)
        print("{} setup done".format(mp.get_start_method()))

    for dir in ["logs", "result"]:
        if not os.path.exists(dir):
            os.makedirs(dir)
        
    ##
    ## for debug: マルチプロセスしない
    ##
    if args.num_processes == 1:
        train(0, args, model, device, kwargs, dataset_larning, dataset_accuracy_train, dataset_accuracy_test)
        exit()

    ##
    ## multiprocessing
    ##
    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank, args, model, device, kwargs, dataset_larning, dataset_accuracy_train, dataset_accuracy_test))
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # save result
    modelname = "result/" + args.log + ".pt"
    model = model.to("cpu")
    torch.save(model.state_dict(), modelname)
    print("save: " + modelname)

    
if __name__ == '__main__':
    main()

# import pdb; pdb.set_trace()

### Local Variables: ###
### truncate-lines:t ###
### End: ###
