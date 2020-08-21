import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from fc import FCNet
import torch.nn.functional as F
import config
import math

class Attention(nn.Module):
    def __init__(self,hidden,mid,dropout):
        super(Attention,self).__init__()
        self.opt=config.parse_opt()
        self.v_proj=nn.Linear(hidden,mid)
        self.q_proj=nn.Linear(hidden,mid)
        self.att=nn.Linear(mid,1)
        self.softmax=nn.Softmax()
        self.dropout=nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self,v,q):
        
        v_proj=self.v_proj(v)
        v_proj=self.dropout(v_proj)
        
        q_proj=torch.unsqueeze(self.q_proj(q),1)
        q_proj=self.dropout(q_proj)
        
        vq_proj=F.relu(v_proj +q_proj)
        #q_proj = torch.unsqueeze(q_proj,1)
        #print q_proj.shape
        #q_proj = q_proj.repeat(1,v.size(-2),1)
        #print q_proj.shape
        #vq_proj=F.relu(torch.cat((v_proj,q_proj),2))
        #print vq_proj.shape
        proj=torch.squeeze(self.att(vq_proj))
        w_att=torch.unsqueeze(self.softmax(proj),2)
        
        vatt = v * w_att
        att_result = torch.squeeze(torch.sum(vatt,1,keepdim=True))
        
        
        #att_result = att_result + self.dropout(sublayer(self.norm(att_result)))
        
        return w_att, att_result

    def self_attention(self,query,key,value):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        #if mask is not None:
        #    scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        #if dropout is not None:
        #    p_attn = dropout(p_attn)
        print p_attn.shape
        print value.shape
        att = torch.matmul(p_attn, value)
        att_result = torch.squeeze(torch.sum(att,1,keepdim=True))
        return att_result, p_attn
