import os
import time 
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import torch.nn.functional as F
import config
import cPickle as pkl
import json

def adjust_learning_rate(optimizer, decay_rate):
	for param_group in optimizer.param_groups:
		param_group['lr'] = param_group['lr'] * decay_rate

def init_model(model):
	for m in model.modules():
		if isinstance(m, nn.Linear):
			nn.init.normal_(m.weight, 0, 0.01)
			m.bias.data.zero_()
			#nn.init.xavier_normal_(m.weight.data)
			#nn.init.constant_(m.bias.data, 0.0)
	return model

def log_hyperpara(logger,opt):
	dic = vars(opt)
	for k,v in dic.items():
		logger.write(k + ' : ' + str(v))
        
def instance_bce_with_logits(logits,labels):
	#logits=torch.round(logits)
	size=logits.shape[0]
	loss=((logits-labels)*(logits-labels)).sum().float()
	#print "True Loss",loss
	return loss

def instance_mse_with_logits(logits,labels):
	#print logits.shape
	#exit(0)
	#logits=torch.clamp(torch.round(logits),1,10)
	logits=torch.clamp(logits,1,10)
	loss=((logits-labels)*(logits-labels)).sum().float()
	return loss

def conv_TF2NUM(x):
	if x == False:
		return 0
	else:
		return 1

def save_info(logits,labels,iter_num,question,gif_name,index,mode):
	logits=logits.cpu()
	logits1=logits.cpu()
	logits=torch.clamp(torch.round(logits),0,10)
	logits=logits.numpy()
	labels=labels.numpy()
	logits1=logits1.numpy()
	#print logits
	#print labels
	score=(logits == labels)
	print score,type(score)
	file_path="./info"
	result_list = []
	#info=''
	#full_path=os.path.join(file_path,mode+'_iter_'+str(iter_num)+'.txt')
	#file_open=open(full_path,'a')
	for i in range(len(score)):
		#print type(logits[i])
		entry={
		'question':question[i],
		'label':int(labels[i]),
		'pred':int(logits[i]),
		'real_pred':float(logits1[i]),            
		'gif_name':gif_name[i],
		'index':index[i],
		'score':conv_TF2NUM(score[i])
		}
		result_list.append(entry)
		#info=info+str(logits[i].numpy())+' '+str(labels[i].numpy())+' '+str(index[i])+' '+str(gif_name[i])+' '+str(question[i])+'\n'
		#dict_info[str(iter_num)]=info
	#file_open.write(info)
	#file_open.close()
	return result_list

def compute_score(logits,labels):
	#criterion=nn.MSELoss(reduce=True, size_average=True)
	#logits=torch.max(logits,1)[1].data
	#labels=torch.max(labels,1)[1]
	result=instance_mse_with_logits(logits,labels)
	#print "Compute score MSE",result
	#print 'Shape:',logits.shape,labels.shape
	score=torch.eq(torch.clamp(torch.round(logits),0,10),labels).sum().float()
	print 'Prediction:',logits
	print logits.shape
	print 'Label:',labels
	print labels.shape
	return score,result

def train(model,train_loader,test_loader,num_epochs,output,opt):
	optim=torch.optim.Adamax(model.parameters(),lr=0.001)
	model = init_model(model)
	#optim = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
	#optim = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99))
	logger=utils.Logger(os.path.join(output,'log'+str(opt.save_num)+'.txt'))
	log_hyperpara(logger,opt)    
	best_eval_loss=100
	for epoch in range(num_epochs):
		total_loss=0
		MSE_Loss=0
		train_score=0
		t=time.time()
		info=''
		for i ,(conct,q,l,ques,g,im) in enumerate(train_loader):
			conct=conct.float().cuda()
			v=conct.float().cuda()
			q=q.cuda()
			l=l.cuda()
			pred=model(v,q)
			#if epoch % opt.DECAY_STEPS == 0 and epoch != 0:
				#adjust_learning_rate(optim,opt.DECAY_RATE)
			#print pred,pred.requires_grad
			loss=instance_bce_with_logits(pred,l)
			#print 'M_loss:',loss,type(loss),loss.requires_grad
			loss.backward()    
			nn.utils.clip_grad_norm(model.parameters(),opt.clip_grad)
			optim.step()
			optim.zero_grad()
			score,MSE=compute_score(pred,l)
			batch_score=score
			total_loss+=loss
			MSE_Loss+=MSE
			train_score+=batch_score

		MSE_Loss=MSE_Loss.float()/len(train_loader.dataset)
		print 'Length of training:',len(train_loader.dataset)
		train_score=100 * train_score.float()/len(train_loader.dataset)
		model.train(False)
        
		evaluate_score,M_l=evaluate(model,test_loader,epoch)
		print 'Epoch:',epoch,'evaluation score:',evaluate_score
		print 'Epoch:',epoch,'MSE of train:',MSE_Loss
		print 'Epoch:',epoch,'MSE of evaluation:',M_l
		model.train(True)

		logger.write('epoch %d, time: %.2f' %(epoch, time.time() -t))
		logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
		logger.write('\teval score: %.2f ' % (100 * evaluate_score))
		logger.write('\tMSE loss of evaluation: %.2f ' % (M_l))
        
		if M_l<best_eval_loss:
			if opt.save == True:            
				model_path=os.path.join(output,str(epoch)+'_model.pth')
				torch.save(model.state_dict(),model_path)
			best_eval_loss=M_l
	logger.write('best loss %.2f' %(best_eval_loss))
            
def evaluate(model,test_loader,epoch):
	score=0
	num_data=0
	count=0
	ml=0
	count_iter=0
	result_list = []
	file_path="./info"
	for conct,q,l,question,gif_name,index in iter(test_loader):
		conct=conct.float().cuda()
		v=Variable(conct,volatile=True).float().cuda()
		q=Variable(q,volatile=True).cuda()
		with torch.no_grad():
			pred=model(v,q)
		score1,MSE=compute_score(pred,l.cuda())
		batch_score=score1.float()
		score=score+batch_score
		num_data+=pred.size(0)
		count+=1
		#batch_MSE=instance_mse_with_logits(pred,l.cuda())
		print 'Batch MSE of evaluation:',MSE
		ml=ml+MSE.float()
		temp_result = save_info(pred,l,epoch,question,gif_name,index,'eval')
		result_list.extend(temp_result)
		print 'Batch score of evaluation is:',batch_score
		count_iter=count_iter+1
	#print result_list
	eval_path=os.path.join(file_path,'Eval_iter_'+str(epoch)+'.json')
	with open(eval_path,'w') as f:
		json.dump(result_list,f)
	score=score/len(test_loader.dataset)
	print 'The length of the loader is:',len(test_loader.dataset)
	return score,ml/len(test_loader.dataset)
