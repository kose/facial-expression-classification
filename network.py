import torch
from torch import nn
from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self, dim1, dim2, dim3):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, dim1, kernel_size=3, padding=1, stride=2) # 64 -> 32
        self.bn1 = nn.BatchNorm2d(dim1)
        self.dropout1 = nn.Dropout2d(p=0.3)

        self.conv2 = nn.Conv2d(dim1, dim2, kernel_size=3, padding=1, stride=2) # 32 -> 16
        self.bn2 = nn.BatchNorm2d(dim2)
        self.dropout2 = nn.Dropout2d(p=0.3)

        self.conv3 = nn.Conv2d(dim2, dim3, kernel_size=3, padding=1, stride=2) # 16 -> 8
        self.bn3 = nn.BatchNorm2d(dim3)
        self.dropout3 = nn.Dropout2d(p=0.4)

        self.conv4 = nn.Conv2d(dim3, dim1, kernel_size=3, padding=1, stride=2) # 8 -> 4
        self.bn4 = nn.BatchNorm2d(dim1)
        self.dropout4 = nn.Dropout2d(p=0.5)

        self.ln1 = nn.Linear(4 * 4 * dim1, 4 * 4 * dim1)
        self.ln2 = nn.Linear(4 * 4 * dim1, 4)

        # initial wait, bias
        for net in [self.conv1, self.conv2, self.conv3, self.conv4, self.ln1, self.ln2]:
            # nn.init.xavier_normal_(net.weight, gain=1.0) #Xavier(正規分布)
            nn.init.kaiming_normal_(net.weight) # He(正規分布)
            nn.init.uniform_(net.bias, 0.0, 1.0) # 一様分布


    def forward(self, x):
        batchsize = x.shape[0]
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.dropout1(h)
        
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.dropout2(h)
        
        h = F.relu(self.bn3(self.conv3(h)))
        h = self.dropout3(h)
        
        h = F.relu(self.bn4(self.conv4(h)))
        h = self.dropout4(h)

        h = h.view(batchsize, -1)
        h = F.relu(self.ln1(h))
        h = self.ln2(h)

        output = F.log_softmax(h, dim=1)
        
        return output

    def loss_function(self, x, labels):
        h = self.forward(x)
        return F.nll_loss(h, labels)

# import pdb; pdb.set_trace()

### Local Variables: ###
### truncate-lines:t ###
### End: ###
