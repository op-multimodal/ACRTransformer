import torch
import torch.nn.functional as F
import torch.nn as nn
from attention import Attention
from language_model import WordEmbedding,QuestionEmbedding
from video_model import VideoEmbedding
from fc import FCNet
from classifier import SimpleClassifier
import config

class BaseModel(nn.Module):
	def __init__(self,w_emb,q_emb,v_att,classifier,v_emb):
		super(BaseModel,self).__init__()
		self.w_emb=w_emb
		self.q_emb=q_emb
		self.v_emb=v_emb
		self.v_att=v_att
		self.classifier=classifier
		self.v_proj=nn.Linear(1024,512)
		self.q_proj=nn.Linear(1024,512)
		self.att=nn.Linear(512,1)
		self.softmax=nn.Softmax()

	def forward(self,v,q):
		self.opt=config.parse_opt()
		w_emb=self.w_emb(q)
		v_embedding=self.v_emb(v)
		q_emb=self.q_emb(w_emb)
		w_att = self.v_att(v_embedding,q_emb)
		#v_proj=self.v_proj(v_embedding)
		#q_proj=torch.unsqueeze(self.q_proj(q_emb),1)
		#vq_proj=F.relu(v_proj +q_proj)
		#proj=torch.squeeze(self.att(vq_proj))
		#w_att=torch.unsqueeze(self.softmax(proj),2)
		v_att=v_embedding * w_att
		vatt=torch.squeeze(torch.sum(v_att,1,keepdim=True))
		joint_proj=vatt+q_emb
		logits=torch.squeeze(self.classifier(joint_proj))
		return logits

def build_baseline(dataset):
	opt=config.parse_opt()
	w_emb=WordEmbedding(dataset.dictionary.ntokens(),300,opt.EMB_DROPOUT)
	q_emb=QuestionEmbedding(300,opt.NUM_HIDDEN,opt.NUM_LAYER,opt.BIDIRECT,opt.L_RNN_DROPOUT)
	v_emb=VideoEmbedding(opt.C3D_SIZE+opt.RES_SIZE,opt.NUM_HIDDEN,opt.NUM_LAYER,opt.BIDIRECT,opt.L_RNN_DROPOUT)
	v_att=Attention(opt.NUM_HIDDEN,opt.MID_DIM,opt.FC_DROPOUT)
	classifier=SimpleClassifier(opt.NUM_HIDDEN,opt.MID_DIM,1,opt.FC_DROPOUT)
	return BaseModel(w_emb,q_emb,v_att,classifier,v_emb)

