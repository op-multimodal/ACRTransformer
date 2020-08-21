import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from fc import FCNet
import torch.nn.functional as F
import config

class Attention(nn.Module):
    def __init__(self,hidden,mid,dropout):
        super(Attention,self).__init__()
        self.opt=config.parse_opt()
        self.v_proj=nn.Linear(hidden,mid)
        self.q_proj=nn.Linear(hidden,mid)
        self.att=nn.Linear(mid,1)
        self.softmax=nn.Softmax()
        self.dropout=nn.Dropout(dropout)


    def forward(self,v,q):
        
        v_proj=self.v_proj(v)
        v_proj=self.dropout(v_proj)
        
        q_proj=torch.unsqueeze(self.q_proj(q),1)
        q_proj=self.dropout(q_proj)
        
        vq_proj=F.relu(v_proj +q_proj)
        proj=torch.squeeze(self.att(vq_proj))
        w_att=torch.unsqueeze(self.softmax(proj),2)
        
        return w_att


