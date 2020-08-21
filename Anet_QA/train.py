import numpy
import os
import time 
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import json
import config
import optimizer
import cPickle as pkl
from calculate_wups import answer_wup
import datetime

def load_pkl(path):
	data=pkl.load(open(path,'r'))
	return data


def log_hyperpara(logger,opt):
	dic = vars(opt)
	for k,v in dic.items():
		logger.write(k + ' : ' + str(v))

def instance_bce_with_logits(logits,labels):
	loss=nn.functional.binary_cross_entropy_with_logits(logits, labels)
	loss*=labels.size(1)
	return loss

def compute_score(pred,ques_id,val_file,ans_file):
	score=0.0
	result=torch.max(pred,dim=1)[1]
	for i,p in enumerate(result):
		answer=val_file[ques_id[i]].strip()
		text_p=ans_file[p].strip()
		if text_p==answer:
			score+=1
	return score


def score_for_anet_wp(pred,ques_id,val_file,ans_file):
	score=0.0
	wup0=0.0
	wup9=0.0
	result=torch.max(pred,dim=1)[1]
	for i,p in enumerate(result):
		answer=val_file[ques_id[i]].strip()
		text_p=ans_file[p].strip()
		wup0+=answer_wup(text_p,answer,0.0)
		wup9+=answer_wup(text_p,answer,0.9)       
		if text_p==answer:
			score+=1
	return score,wup0,wup9

def train(model,train_loader,test_loader,opt):
	opt=config.parse_opt()
	optim=optimizer.get_std_opt(model, opt)
	num_epochs=opt.EPOCHS
	#optim=torch.optim.Adamax(model.parameters())
	logger=utils.Logger(os.path.join('./save_models','log'+str(opt.SAVE_NUM)+'.txt'))
	dict_file=load_pkl(opt.ANET_LABEL2ANS)
	log_hyperpara(logger,opt)
	best_eval_score=0
	train_file=load_pkl(opt.ANET_TRAIN_DICT)
    
	for epoch in range(num_epochs):            
		total_loss=0
		train_score=0.0
		t=time.time()
		t_time=0.0           
		for i ,(conct,q,l,ques_id) in enumerate(train_loader):
			starttime = datetime.datetime.now()             
			conct=conct.float().cuda()
			v=Variable(conct).float().cuda()
			q=Variable(q).cuda()
			l=Variable(l).float().cuda()
			pred=model(v,q)
			loss=instance_bce_with_logits(pred,l.cuda())
			loss.backward()
			nn.utils.clip_grad_norm(model.parameters(),0.25)
			optim.step()
			optim.zero_grad()
			batch_score=compute_score(pred,ques_id,train_file,dict_file)
			total_loss+=loss*v.size(0)
			train_score+=batch_score
			print 'Epoch:',epoch,'batch:',i+1,'bathc_score:',batch_score,'loss:',loss
			endtime = datetime.datetime.now() 
			t_time+=(endtime-starttime).microseconds          
		logger.write('epoch %d, time: %.2f' %(epoch, time.time() -t))
		logger.write('time cost: %.2f' %(t_time/1000000))
		total_loss/=len(train_loader.dataset)
		train_score=100 * train_score/len(train_loader.dataset)
		model.train(False)       
		evaluate_score,test_loss,w0,w9=evaluate(model,test_loader,opt,epoch)
		w0=w0*100
		w9=w9*100
		print 'Epoch:',epoch,'evaluation w0:',w0,' w9:',w9
		logger.write('\twup0: %.2f, wup9: %.2f' % (w0, w9))       
		print 'Epoch:',epoch,'evaluation score:',100*evaluate_score,' loss:',test_loss
		logger.write('\ttrain_loss: %.2f, accuracy: %.2f' % (total_loss, train_score))
		logger.write('\teval accuracy: %.2f ' % (100 * evaluate_score))
		logger.write('\teval loss: %.2f ' % (test_loss))
		if evaluate_score>best_eval_score:
			best_eval_score=evaluate_score
		model.train(True)
	logger.write('best accuracy %.2f' %(100*best_eval_score))
        


def evaluate(model,test_loader,opt,epoch):
	score=0.0
	total_loss=0.0
	total_p0=0.0
	total_p9=0.0
	val_file=load_pkl(opt.ANET_VAL_DICT)
	dict_file=load_pkl(opt.ANET_LABEL2ANS)    
	for i,(feat,q_token,label,ques_id) in enumerate(test_loader):
		with torch.no_grad():
			v=feat.float().cuda()
			q=q_token.cuda()
			label=label.float().cuda()
			pred=model(v,q)
		batch_score,p0,p9=score_for_anet_wp(pred,ques_id,val_file,dict_file)
		total_p0+=p0
		total_p9+=p9
		loss=instance_bce_with_logits(pred,label)
		total_loss+= loss       
		score+=batch_score        
	score=score/len(test_loader.dataset)
	avg_loss=total_loss/len(test_loader.dataset) 
	total_p0=total_p0/len(test_loader.dataset)
	total_p9=total_p9/len(test_loader.dataset)   
	return score,avg_loss,total_p0,total_p9  
