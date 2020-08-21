import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Base_Op,Activity_Op,TGIF_Op,TVQA_Op,Activity_Op,TVQA_Iter_Op
import baseline
from train import train
import utils
import config

if __name__=='__main__':
    opt=config.parse_opt()
    torch.cuda.set_device(opt.CUDA_DEVICE)
    torch.manual_seed(opt.SEED)
    
    dictionary=Base_Op()
    dictionary.init_dict()
    
    if opt.DATASET=='tgif-qa':
        train_set=TGIF_Op(opt,dictionary,'Train')
        test_set=TGIF_Op(opt,dictionary,'Test')
        constructor='build_baseline'
        model=getattr(baseline,constructor)(train_set,opt).cuda()
        model.w_emb.init_embedding()
        train_loader=DataLoader(train_set,opt.BATCH_SIZE,shuffle=True,num_workers=1)
        test_loader=DataLoader(test_set,opt.BATCH_SIZE,shuffle=True,num_workers=1)
        print 'Length of train:',len(train_loader)
        print 'Length of test:',len(test_loader)
        train_for_tgif(model,train_loader,test_loader,opt)
        exit(0)
    elif opt.DATASET=='tvqa' and opt.USE_ITER==False:
        '''use the same code of tgif-qa with config QUES_TYPE='Action' or 'Trans' '''
        train_set=TVQA_Op(opt,dictionary,'train')
        test_set=TVQA_Op(opt,dictionary,'val')
        constructor='build_baseline'
        model=getattr(baseline,constructor)(train_set,opt).cuda()
        model.w_emb.init_embedding()
        train_loader=DataLoader(train_set,opt.BATCH_SIZE,shuffle=True,num_workers=1)
        test_loader=DataLoader(test_set,opt.BATCH_SIZE,shuffle=True,num_workers=1)
        print 'Length of train:',len(train_loader)
        print 'Length of test:',len(test_loader)
        train_for_tgif(model,train_loader,test_loader,opt)
        exit(0)  
    elif opt.DATASET=='tvqa' and opt.USE_ITER==True:
        train_set=TVQA_Iter_Op(opt,dictionary,'train')
        test_set=TVQA_Op(opt,dictionary,'val')        
        constructor='build_baseline'
        model=getattr(baseline,constructor)(train_set,opt).cuda()
        model.w_emb.init_embedding() 
        test_loader=DataLoader(test_set,opt.BATCH_SIZE,shuffle=True,num_workers=1)
        train_for_iter_tvqa(model,test_loader,opt,train_set)
        exit(0)
    elif opt.DATASET=='anet-qa':
        train_set=Activity_Op(opt,dictionary,'train')
        val_set=Activity_Op(opt,dictionary,'test')
        constructor='build_baseline'
        model=getattr(baseline,constructor)(train_set,opt).cuda()
        model.w_emb.init_embedding()
        train_loader=DataLoader(train_set,opt.BATCH_SIZE,shuffle=True,num_workers=1)
        val_loader=DataLoader(val_set,opt.BATCH_SIZE,shuffle=True,num_workers=1)     
        print 'Length of train:',len(train_loader)
        print 'Length of test:',len(val_loader)
        train(model,train_loader,val_loader,opt)
        exit(0)        
        
        