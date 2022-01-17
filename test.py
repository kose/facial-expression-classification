#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os
import numpy as np
from sklearn.metrics import confusion_matrix

import torch
from torchvision import datasets, transforms
from dataset import MyDataset, Flip
from network import Net

dictionary = {0:"neutral", 1:"happy", 2:"sad", 3:"angry"}

##
## make csv file
##
def make_result(args, model, device):

    dataset = MyDataset(csvfile="test_master.tsv", flip=Flip.none, transform=False, repeat=1)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=False)

    df = dataset.df
    data, labels_ground_truth = testloader.__iter__().next()
    
    labels_ground_truth = labels_ground_truth.to("cpu").numpy().copy()

    _pred = model(data).to("cpu").numpy().copy()
    labels_pred = np.argmax(_pred, axis=1)

    with open(args.csv, "w") as f:
        for i in range(len(labels_pred)):
            filename = os.path.basename(df.id[i])
            sr = filename + "," + str(dictionary[labels_pred[i]]) + "\n"
            f.writelines(sr)

##
## test main function
##
def test(args, model, device):

    dataset = MyDataset(csvfile="train_master.tsv", flip=Flip.both, transform=False, repeat=1)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=False)
    
    data, labels_ground_truth = testloader.__iter__().next()
    
    labels_ground_truth = labels_ground_truth.numpy().copy()

    _pred = model(data).numpy().copy()
    labels_pred = np.argmax(_pred, axis=1)

    result = confusion_matrix(labels_ground_truth, labels_pred)

    print(result)

    ## accuracy

    accuracy = sum(labels_ground_truth == labels_pred) / len(labels_ground_truth)
    print("accuracy:", accuracy)
    
    # print("end")


##
## main function
##
def main():
# Testing settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--model', type=str, default="result/result.pt")
    parser.add_argument('--csv', type=str, default="result/result.csv")
    parser.add_argument('--dim1', type=int, default=32, 
                        help='Neuralnetwork dimension 1 (default: 32)')
    parser.add_argument('--dim2', type=int, default=64, 
                        help='Neuralnetwork dimension 2 (default: 64)')
    parser.add_argument('--dim3', type=int, default=64, 
                        help='Neuralnetwork dimension 3 (default: 64)')

    args = parser.parse_args()

    # modelname = os.path.join("result", args.model)
    modelname = args.model

    print(modelname)
    
    device = torch.device("cpu")

    model = Net(dim1=args.dim1, dim2=args.dim2, dim3=args.dim3)

    model.load_state_dict(torch.load(modelname))

    #
    with torch.no_grad():
        model.eval()
        test(args, model, device)
        make_result(args, model, device)


if __name__ == '__main__':
    main()

# import pdb; pdb.set_trace()

### Local Variables: ###
### truncate-lines:t ###
### End: ###
