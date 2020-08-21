import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
import config

class Gate_combine(nn.Module):
    def __init__(self,hidden,mid,dropout):
        super(Gate_combine,self).__init__()
        self.opt=config.parse_opt()
        
        self.a_proj=nn.Linear(hidden,mid)
        self.a_att=nn.Linear(mid,1)
        self.f_proj=nn.Linear(hidden,mid)
        self.f_att=nn.Linear(mid,1)
        self.q_proj=nn.Linear(hidden,mid)
        self.q_att=nn.Linear(mid,1)
        
        self.sig=nn.Sigmoid()
        self.dropout=nn.Dropout(dropout)


    def forward(self,f,a,q):
        
        f_proj=self.f_proj(f+q)
        f_proj=self.dropout(f_proj)
        f_g = self.sig(self.f_att(f_proj))
        
        a_proj=self.a_proj(a+q)
        a_proj=self.dropout(a_proj)
        a_g = self.sig(self.a_att(a_proj))
        
        q_proj=self.q_proj(q)
        q_proj=self.dropout(q_proj)
        q_g = self.sig(self.q_att(q_proj))
        
        faq_comb=f_g*f+a_g*a + q_g*q
        
        return faq_comb


