import torch
import torch.nn as nn
import config
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F

class MFB(nn.Module):
	def __init__(self):
		super(MFB,self).__init__()
		self.opt=config.parse_opt()
		self.dropout=nn.Dropout(self.opt.MFH_DROPOUT)
                self.v_proj=nn.Linear(self.opt.NUM_GLIMPSE*self.opt.NUM_HIDDEN,self.opt.MFB_DIM * self.opt.POOLING_SIZE)
                self.q_proj=nn.Linear(self.opt.NUM_HIDDEN,self.opt.MFB_DIM * self.opt.POOLING_SIZE)
		self.v_proj1=nn.Linear(self.opt.NUM_GLIMPSE*self.opt.NUM_HIDDEN,self.opt.MFB_DIM * self.opt.POOLING_SIZE)
                self.q_proj1=nn.Linear(self.opt.NUM_HIDDEN,self.opt.MFB_DIM * self.opt.POOLING_SIZE)
	def forward(self,v,q):
		#round 1
		v_proj1=self.v_proj(v)
		q_proj1=self.q_proj(q)
		vq_elm1=torch.mul(v_proj1,q_proj1)
		vq_drop1=self.dropout(vq_elm1)
		vq_resh1=vq_drop1.view(self.opt.BATCH_SIZE,1,self.opt.MFB_DIM,self.opt.POOLING_SIZE)
		vq_sumpool1=torch.sum(vq_resh1,3,keepdim=True)
		vq_out1=torch.squeeze(vq_sumpool1)
		vq_norm1=torch.sqrt(F.relu(vq_out1))-torch.sqrt(F.relu(-vq_out1))
		mfb1=F.normalize(vq_norm1)
		
		#round 2
		v_proj2=self.v_proj1(v)
                q_proj2=self.q_proj1(q)
                vq_elm2=torch.mul(v_proj2,q_proj2)
                vq_drop2=self.dropout(vq_elm2)
                vq_resh2=vq_drop2.view(self.opt.BATCH_SIZE,1,self.opt.MFB_DIM,self.opt.POOLING_SIZE)
                vq_sumpool2=torch.sum(vq_resh2,3,keepdim=True)
                vq_out2=torch.squeeze(vq_sumpool2)
                vq_norm2=torch.sqrt(F.relu(vq_out2))-torch.sqrt(F.relu(-vq_out2))
                mfb2=F.normalize(vq_norm2)
		
		mfb_f=torch.cat((mfb1,mfb2),1)
		return mfb_f

