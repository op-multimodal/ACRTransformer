import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

class FCNet(nn.Module):
    def __init__(self,in_dim,out_dim,dropout):
        super(FCNet,self).__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.relu=nn.ReLU()
        self.linear=weight_norm(nn.Linear(in_dim,out_dim),dim=None)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,x):
        logits=self.dropout(self.linear(x))
        return logits
        