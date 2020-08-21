import torch
import torch.nn as nn
from attention import Attention
from language_model import WordEmbedding,QuestionEmbedding
import torch.nn.functional as F
from video_model import VideoEmbedding,Videofc
from fc import FCNet
from classifier import SimpleClassifier
from answer_model import AnswerEmbedding
from rela_model import Rela_Module
import config
import numpy as np
from ques_att import Q_Att

class BaseModel(nn.Module):
    def __init__(self,w_emb,q_emb,v_emb,a_emb,v_att,v_fc,r_emb,r_att,classifier,ques_att,opt):
        super(BaseModel,self).__init__()
        self.opt=opt
        self.w_emb=w_emb
        self.q_emb=q_emb
        self.v_emb=v_emb
        self.a_emb=a_emb
        self.v_att=v_att
        self.r_att = r_att
        self.v_fc = v_fc
        self.r_emb = r_emb
        self.ques_att = ques_att
        #self.vlinear = nn.Linear(self.opt.NUM_HIDDEN,self.opt.NUM_HIDDEN)
        #self.rlinear = nn.Linear(self.opt.NUM_HIDDEN,self.opt.NUM_HIDDEN)
        self.classifier = classifier
        self.softmax=nn.Softmax()

    def forward(self,v,q):
        w_emb=self.w_emb(q)
        
        #q_emball=self.q_emb.forward_all(w_emb)
        #q_emb,_ = self.v_att.self_attention(q_emball,q_emball,q_emball)
        
        #question = self.q_emb.forward_all(w_emb)
        #q_emb = self.ques_att(question)
        
        q_emb = self.q_emb(w_emb)
        
        v_emb = self.v_fc(v)
        
        r_emb = self.r_emb(v_emb,q_emb)
        
        w_att, vatt = self.v_att(v_emb,q_emb)

        wr_att,ratt = self.r_att(r_emb,q_emb)
        
        #vatt = self.vlinear(vatt)
        #ratt = self.rlinear(ratt)
        
        joint_proj = vatt + q_emb
        joint_proj = joint_proj + ratt
        joint_proj = torch.cat((joint_proj,q_emb),1)
        logits= torch.squeeze(self.classifier(joint_proj))
        
        #logits=torch.round(logits)
        return logits

def build_baseline(dataset,opt):
    w_emb=WordEmbedding(dataset.dictionary.ntokens(),300,opt.EMB_DROPOUT)
    q_emb=QuestionEmbedding(300,opt.NUM_HIDDEN,opt.NUM_LAYER,opt.BIDIRECT,opt.L_RNN_DROPOUT)
    v_emb=VideoEmbedding(opt.C3D_SIZE+opt.RES_SIZE,opt.NUM_HIDDEN,opt.NUM_LAYER,opt.BIDIRECT,opt.L_RNN_DROPOUT)
    v_att=Attention(opt.NUM_HIDDEN,opt.MID_DIM,opt.FC_DROPOUT)
    r_att=Attention(opt.NUM_HIDDEN,opt.MID_DIM,opt.FC_DROPOUT)
    v_fc=Videofc(opt.GLIMPSE,opt.C3D_SIZE+opt.RES_SIZE,opt.NUM_HIDDEN,opt.FC_DROPOUT)
    a_emb=AnswerEmbedding(300,opt.NUM_HIDDEN,opt.NUM_LAYER,opt.BIDIRECT,opt.L_RNN_DROPOUT)
    rela_emb = Rela_Module(opt.NUM_HIDDEN*3,opt.NUM_HIDDEN,opt.NUM_HIDDEN)
    classifier=SimpleClassifier(opt.NUM_HIDDEN*2,opt.MID_DIM,1,opt.FC_DROPOUT)
    ques_att = Q_Att(opt.NUM_HIDDEN,opt.MID_DIM,opt.FC_DROPOUT)
    #vlinear=FCNet([opt.NUM_HIDDEN,opt.MID_DIM,opt.NUM_HIDDEN])
    #rlinear=FCNet([opt.NUM_HIDDEN,opt.MID_DIM,opt.NUM_HIDDEN])
    
    return BaseModel(w_emb,q_emb,v_emb,a_emb,v_att,v_fc,rela_emb,r_att,classifier,ques_att,opt)

