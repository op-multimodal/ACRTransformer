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

def log_hyperpara(logger,opt):
	dic = vars(opt)
	for k,v in dic.items():
		logger.write(k + ' : ' + str(v))

def instance_bce_with_logits(logits,labels):
	loss=nn.functional.binary_cross_entropy_with_logits(logits, labels)
	loss*=labels.size(1)
	return loss

def compute_score(logits,labels):
	logits=torch.max(logits,1)[1].cuda()
	one_hot=torch.zeros(*labels.size()).cuda()
	one_hot.scatter_(1,logits.view(-1,1),1)
	score=(one_hot * labels)
#	print type(score),score
#	print 'Prediction:',logits
	#print 'Label:',torch.max(labels,1)[1]
	return score

def save_info(logits,labels,iter_num,question,gif_name,index):
	logits=torch.max(logits,1)[1].cpu()
	#one_hot=torch.zeros(*labels.size()).cuda()
	#one_hot.scatter_(1,logits.view(-1,1),1)
	labels=torch.max(labels,1)[1]
	score=(logits == labels)
	label2ans=pkl.load(open("./data/label2ans.pkl",'r'))
	print score,type(score)
	file_path="./info"
	#dict_info={}
	info=''
	full_path=os.path.join(file_path,'iter_'+str(iter_num)+'.txt')
	file_open=open(full_path,'a')
	for i in range(len(score)):
		if score[i]==0:
			info=info+str(label2ans[logits[i]])+' '+str(label2ans[labels[i]])+' '+str(index[i])+' '+str(gif_name[i])+' '+str(question[i])+'\n'
			#dict_info[str(iter_num)]=info
	file_open.write(info)
	file_open.close()
	

def train(model,train_loader,test_loader,num_epochs,output,opt):
	#opt=config.parse_opt() 
	#optim=optimizer.get_std_opt(model, opt)
	optim=torch.optim.Adamax(model.parameters())
	logger=utils.Logger(os.path.join(output,'log'+str(opt.save_num)+'.txt'))
	#logger=utils.Logger(os.path.join(output,'log.txt'))
	log_hyperpara(logger,opt)
	best_eval_score=0
    
	for epoch in range(num_epochs):           
		total_loss=0
		train_score=0
		t=time.time()
		for i ,(conct,q,l,ques,g,im) in enumerate(train_loader):
			conct=conct.float().cuda()
			v=Variable(conct).float().cuda()
			q=Variable(q).cuda()
			l=Variable(l).float().cuda()
			pred=model(v,q)
			#print pred
			loss=instance_bce_with_logits(pred,l.cuda())
			loss.backward()
			#nn.utils.clip_grad_norm(model.parameters(),0.25)
			optim.step()
			#optim.optimizer.zero_grad()
			optim.zero_grad()
			batch_score=compute_score(pred,l.data).sum()
			total_loss+=loss*v.size(0)
			train_score+=batch_score
			print 'Epoch:',epoch,'batch:',i+1,'bathc_score:',batch_score,'loss:',loss
		total_loss/=len(train_loader.dataset)
		train_score=100 * train_score/len(train_loader.dataset)
		model.train(False)
		evaluate_score=evaluate(model,test_loader,epoch)
		print 'Epoch:',epoch,'evaluation score:',evaluate_score
		model.train(True)

		logger.write('epoch %d, time: %.2f' %(epoch, time.time() -t))
		logger.write('\ttrain_loss: %.2f, accuracy: %.2f' % (total_loss, train_score))
		logger.write('\teval accuracy: %.2f ' % (100 * evaluate_score))
        
		if evaluate_score>best_eval_score:
			if opt.save == True:
				model_path=os.path.join(output,str(epoch)+'_model.pth')
				torch.save(model.state_dict(),model_path)
			best_eval_score=evaluate_score
	logger.write('best accuracy %.2f' %(100*best_eval_score))

def evaluate(model,test_loader,epoch):
	score=0
	count_iter=0
	for conct,q,l,question,gif_name,index in iter(test_loader):
		conct=conct.float().cuda()
		v=Variable(conct,volatile=True).float().cuda()
		q=Variable(q,volatile=True).cuda()
		pred=model(v,q)
		batch_score=compute_score(pred,l.cuda()).sum()
		score+=batch_score
		print 'Batch score of evaluation is:',batch_score
		save_info(pred,l,epoch,question,gif_name,index)
		count_iter=count_iter+1
	score=score.float()/len(test_loader.dataset)
	return score
