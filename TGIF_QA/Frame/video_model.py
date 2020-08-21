import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import config

class WordEmbedding(nn.Module):
    def __init__(self,ntoken,emb_dim,dropout):
        super(WordEmbedding,self).__init__()
        self.emb=nn.Embedding(ntoken+1,emb_dim,padding_idx=ntoken)
        self.dropout=nn.Dropout(dropout)
        self.ntoken=ntoken
        self.emb_dim=emb_dim

    def init_embedding(self):
        glove_weight=torch.from_numpy(np.load('./data/glove_embedding.npy'))
        self.emb.weight.data[:self.ntoken]=glove_weight

    def forward(self,x):
        emb=self.emb(x)
        emb=self.dropout(x)
        return emb


class VideoEmbedding(nn.Module):
    def __init__(self,in_dim,num_hidden,num_layer,bidirect,dropout,rnn_type='GRU'):
        super(VideoEmbedding,self).__init__()
        rnn_cls=nn.LSTM if rnn_type=='LSTM' else nn.GRU
        self.rnn=rnn_cls(in_dim,num_hidden,num_layer,bidirectional=bidirect,dropout=dropout,batch_first=True)
        self.in_dim=in_dim
        self.num_hidden=num_hidden
        self.num_layer=num_layer
        self.rnn_type=rnn_type
        self.num_bidirect=1+int(bidirect)

    def init_hidden(self,batch):
        weight=next(self.parameters()).data
        hid_shape=(self.num_layer * self.num_bidirect,batch,self.num_hidden)
        if self.rnn_type =='LSTM':
            return (Variable(weight.new(*hid_shape).zero_()).cuda(),
                    Variable(weight.new(*hid_shape).zero_())).cuda()
        else:
            return Variable(weight.new(*hid_shape).zero_()).cuda()

    def forward_all(self,x):
        batch=x.size(0)
        hidden=self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output,hidden=self.rnn(x,hidden)
        if self.num_bidirect==1:
            return output[:,-1]
        forward_=output[:,-1,self.num_hidden]
        backward_=output[:,0,self.num_hidden]
        return torch.cat((forward_,backward_),dim=1)

    def forward(self,x):
        batch=x.size(0)
        hidden=self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output,hidden=self.rnn(x,hidden)
        return output

class Videofc(nn.Module):
    def __init__(self,glimpse,in_dim,hidden,dropout):
        super(Videofc,self).__init__()
        v_proj = []
        for i in range(glimpse):
            v_proj.append(nn.Linear(in_dim,hidden))
        self.v_proj = nn.ModuleList(v_proj)
        self.vemb_proj = nn.Linear(hidden*glimpse,hidden)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        v_emb_list = []
        v_emb_list = [v_proj_t(x) for v_proj_t in self.v_proj]
        v_emb = torch.cat(v_emb_list,dim=2)
        v_emb = self.vemb_proj(v_emb)
        v_emb = self.dropout(v_emb)
        return v_emb