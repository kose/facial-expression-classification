import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from sklearn.metrics import confusion_matrix

from torch.utils.tensorboard import SummaryWriter

from dataset import MyDataset, Flip

def train(rank, args, model, device, dataset, dataloader_kwargs):

    if rank == 0:
        writer = SummaryWriter("logs/" + args.log)
        accuracy_train_dataset = MyDataset(csvfile="train.tsv", flip=Flip.both, transform=False, repeat=1)
        accuracy_train_loader = torch.utils.data.DataLoader(accuracy_train_dataset, batch_size=10000, shuffle=False)

        accuracy_test_dataset = MyDataset(csvfile="test.tsv", flip=Flip.both, transform=False, repeat=1)
        accuracy_test_loader = torch.utils.data.DataLoader(accuracy_test_dataset, batch_size=10000, shuffle=False)
    else:
        writer = None
    
    torch.manual_seed(args.seed + rank)

    train_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_epoch(rank, epoch, args, model, device, train_loader, optimizer, writer)

        #
        # accuracy
        #
        if rank == 0:
            model.eval()

            ##
            ## training
            ##
            with torch.no_grad():
                data, labels_ground_truth = accuracy_train_loader.__iter__().next()
                data = data.to(device)

                labels_ground_truth = labels_ground_truth.numpy().copy()
                
                _pred = model(data).to("cpu").numpy().copy()
                labels_pred = np.argmax(_pred, axis=1)

                result = confusion_matrix(labels_ground_truth, labels_pred)
                print(result)

                accuracy = sum(labels_ground_truth == labels_pred) / len(labels_ground_truth)
            
                writer.add_scalar("training/accuracy", accuracy, epoch)

                print("training/accuracy", accuracy)

            ##
            ## testing
            ##
            with torch.no_grad():
                data, labels_ground_truth = accuracy_test_loader.__iter__().next()
                data = data.to(device)

                labels_ground_truth = labels_ground_truth.numpy().copy()

                _pred = model(data).to("cpu").numpy().copy()
                labels_pred = np.argmax(_pred, axis=1)

                result = confusion_matrix(labels_ground_truth, labels_pred)
                print(result)

                accuracy = sum(labels_ground_truth == labels_pred) / len(labels_ground_truth)
            
                writer.add_scalar("testing/accuracy", accuracy, epoch)

                print("testing/accuracy", accuracy)


def train_epoch(rank, epoch, args, model, device, data_loader, optimizer, writer):

    model.train()
    pid = os.getpid()

    for batch_idx, (data, labels) in enumerate(data_loader):
        optimizer.zero_grad()
        data, labels = data.to(device), labels.to(device)
        loss = model.loss_function(data, labels)
        loss.backward()
        optimizer.step()

        #
        # tensorboard
        #
        if writer is not None:
            writer.add_scalar("training/loss", loss.item(), 
                              epoch + batch_idx * len(data) / len(data_loader.dataset))

        #
        # tty log
        #
        if batch_idx % args.log_interval == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                100. * batch_idx / len(data_loader), loss.item()))
            if args.dry_run:
                break


def test(args, model, device, dataset, dataloader_kwargs):


    torch.manual_seed(args.seed)

    test_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)

    model.to("cpu")
    
    model.eval()

    with torch.no_grad():
        data, labels_ground_truth = test_loader.__iter__().next()

        labels_ground_truth = labels_ground_truth.numpy().copy()

        _pred = model(data).numpy().copy()
        labels_pred = np.argmax(_pred, axis=1)

        result = confusion_matrix(labels_ground_truth, labels_pred)
        print(result)
    
        # import pdb; pdb.set_trace()
        # print("end")

### Local Variables: ###
### truncate-lines:t ###
### End: ###
