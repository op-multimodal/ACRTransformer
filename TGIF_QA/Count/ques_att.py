import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from fc import FCNet
import torch.nn.functional as F
import config

class Q_Att(nn.Module):
    def __init__(self,hid,mid,dropout):
        super(Q_Att,self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.conv1=nn.Conv2d(hid,mid,1)
        self.conv2=nn.Conv2d(mid,1,1)
        self.softmax=nn.Softmax(dim=1)
            
    def forward(self,ques):
        q_drop=self.dropout(ques)
        q_resh=torch.unsqueeze(q_drop,dim=3).transpose(1,2)
        q_conv1=F.relu(self.conv1(q_resh))
        q_conv2=torch.squeeze(self.conv2(q_conv1))
        q_weight=torch.unsqueeze(self.softmax(q_conv2),dim=2)
        q_emb=q_weight * ques
        q_emb=torch.squeeze(torch.sum(q_emb,dim=1))
        return q_emb
        
