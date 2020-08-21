import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import config
import torch.nn.functional as F
import copy

class rela_embedding(nn.Module):
    def __init__(self, indim, outdim, dropout):
        super(rela_embedding,self).__init__()
        self.g_fc1 = nn.Linear(indim,outdim)
        #self.g_fc2 = nn.Linear(hidden,outdim)
        self.dropout=nn.Dropout(dropout)
       
    def forward(self,img,qst):
        qst = torch.unsqueeze(qst,1)
        qst = qst.repeat(1,34,1)

        x_i = img[:,0:34,:]
        x_j = img[:,1:35,:]
        x_con = torch.cat([x_i,x_j,qst],2)
        
        rela = self.g_fc1(x_con)
        rela = self.dropout(F.relu(rela))
        
        return rela

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff,d_out, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Rela_Module(nn.Module):
    def __init__(self, indim, hiddim, outdim, dropout=0.3):
        super(Rela_Module, self).__init__()
        '''
        ## Variables:
        - indim: dimensionality of input node features
        - hiddim: dimensionality of the joint hidden embedding
        - outdim: dimensionality of the output node features
        - combined_feature_dim: dimensionality of the joint hidden embedding for graph
        - K: number of graph nodes/objects on the image
        '''
        self.in_dim = indim
        self.hid_dim = hiddim
        self.out_dim = outdim
        self.dropout = dropout
        
        self.h = 2
        self.d_k = self.out_dim//self.h
        
        self.rela_node = rela_embedding(self.in_dim, self.out_dim, self.dropout)
        
        self.rela = clones(rela_embedding(self.in_dim, self.d_k, self.dropout), self.h)
        
        self.feed_foward = PositionwiseFeedForward(self.out_dim, self.hid_dim, self.out_dim, self.dropout)
        self.norm = LayerNorm(self.out_dim)

    def forward(self, img, qst):
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        ## Returns:
        - graph_encode_features (batch_size, K, out_feat_dim)
        '''
        r_nodes = self.rela_node(img, qst)
        
        r_feature = \
            tuple([l(img,qst) for l in self.rela])
        
        r_feature = self.norm(torch.cat(r_feature,2)) + r_nodes
        print('r_feature')
        print(r_feature.shape)
        
        rela_encode_features = self.feed_foward(r_feature) + r_feature
        
        return rela_encode_features