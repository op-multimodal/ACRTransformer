import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from dataset import Dictionary,FeatureDataset
import baseline
from train import train
import utils
import config
import os

dirs_list = ['./info','./save_models']

if __name__=='__main__':
	opt=config.parse_opt()
	torch.cuda.set_device(1)
	torch.manual_seed(opt.SEED)
	torch.cuda.manual_seed(opt.SEED)
	torch.backends.cudnn.bechmark=True

	dictionary=Dictionary({'Yes':0},['Yes'])
	dictionary.init_dict()
    
	train_set=FeatureDataset('Action',dictionary,'Train')
	test_set=FeatureDataset('Action',dictionary,'Test')
	constructor='build_baseline'
	model=getattr(baseline,constructor)(train_set,opt).cuda()
	model.w_emb.init_embedding()

	train_loader=DataLoader(train_set,opt.BATCH_SIZE,shuffle=True,num_workers=1)
	test_loader=DataLoader(test_set,opt.BATCH_SIZE,shuffle=True,num_workers=1)
	print 'Length of train:',len(train_loader)
	print 'Length of test:',len(test_loader)
	train(model,train_loader,test_loader,opt.EPOCHS,opt.OUTPUT,opt)
	exit(0)



