import os
import pandas as pd
import json
import cPickle as pkl
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import utils
from tqdm import tqdm
import config
import re
import itertools
import random


'''File related'''
def load_pkl(path):
    data=pkl.load(open(path,'r'))
    return data
    
def read_hdf5(path):
    data=h5py.File(path,'r')
    return data

def read_csv(path):
    data=pd.read_csv(path,sep='\t')
    return data
    
def dump_pkl(path,info):
    pkl.dump(info,open(path,'wb'))  
    
def read_json(path):
    utils.assert_exits(path)
    data=json.load(open(path,'r'))
    '''in anet-qa returns a list'''
    return data
def read_jsonl(path):
    total_info=[]
    with open(path,'r')as f:
        d=f.readlines()
    for i,info in enumerate(d):
        data=json.loads(info)
        total_info.append(data)
    return total_info

class Base_Op(object):
    def __init__(self):
        self.opt=config.parse_opt()
        self.CSV_TYPE={'FrameQA': '_frameqa_question.csv',
          'Count': '_count_question.csv',
          'Trans': '_transition_question.csv',
          'Action' : '_action_question.csv'
         }
    
    def count_answer(self,min_occurence):
        occurence={}
        all_ans=[]
        if self.opt.DATASET=='anet-qa':
            train_path=os.path.join(self.opt.ANET_TEXT_DIR,'train_a.json')
            train_info=read_json(train_path)
            for i,row in enumerate(train_info):
                ans=[row['answer']]
                all_ans.extend(ans)
        elif self.opt.DATASET=='tgif-qa':
            train_path=os.path.join(self.opt.TEXT_DIR,'Train'+self.CSV_TYPE[self.opt.QUES_TYPE])
            train_info=read_csv(train_path).set_index('vid_id')
            for row in train_info.iterrows():
                ans=row[1]['answer'].replace(',','').split(' ')
                all_ans.extend(ans)
            print 'Manipulating on',len(all_ans),'answers'
        for i,ans in enumerate(all_ans):
            if ans not in occurence:
                occurence[ans]=1
            else:
                occurence[ans]+=1
        if self.opt.DATASET=='tgif-qa':     
            for ans in occurence.keys():
                if occurence[ans]<min_occurence:
                    occurence.pop(ans)
            return occurence        
        elif self.opt.DATASET=='anet-qa':
            final=sorted(occurence.items(), key=lambda d:d[1], reverse = True)[:self.opt.VOC_SIZE]
            return final
   
    
    def create_ans2label(self):
        occurence=self.count_answer(self.opt.MIN_OCC)
        print 'The number of answers is:',len(occurence)
        self.ans2label={}
        self.label2ans=[]
        label=0
        if self.opt.DATASET=='tgif-qa':
            for answer in occurence:
                self.label2ans.append(answer)
                self.ans2label[answer]=label
                label+=1
        elif self.opt.DATASET=='anet-qa':
            for i,answer in enumerate(occurence):
                self.label2ans.append(answer[0])
                self.ans2label[answer[0]]=i
        self.label2ans.append('UNK')
        self.ans2label['UNK']=label
        if self.opt.DATASET=='tgif-qa':
            dump_pkl(self.opt.ANS2LABEL,self.ans2label)
            dump_pkl(self.opt.LABEL2ANS,self.label2ans)   
        elif self.opt.DATASET=='anet-qa':
            dump_pkl(self.opt.ANET_ANS2LABEL,self.ans2label)
            dump_pkl(self.opt.ANET_LABEL2ANS,self.label2ans)               
    
    def add_word(self,word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word]=len(self.idx2word)-1
    
    def tokenize(self,sent,add_to_dict):
        sentence=sent.lower()
        sentence=sentence.replace(',','').replace('?','').replace('\'s','\'s').replace('.','')
        words=sentence.split()
        tokens=[]
        if add_to_dict:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                if w in self.word2idx:
                    tokens.append(self.word2idx[w])
                else:
                    tokens.append(self.word2idx['UNK'])
        return tokens
    
    def tokenize_tvqa(self,sent,add_to_dict):
        sentence=sent.lower()
        sentence=sentence.replace(',','').replace('?','').replace('.','')
        words=sentence.split()
        tokens=[]
        if add_to_dict:
            for w in words:
                if w in self.word_count:
                    self.word_count[w]+=1
                else:
                    self.word_count[w]=1
        else:
            for w in words:
                if w in self.word2idx:
                    tokens.append(self.word2idx[w])
                else:
                    tokens.append(self.word2idx['UNK'])
        return tokens    
    
    def dict_token_sent(self):
        train_path=os.path.join(self.opt.ANET_TEXT_DIR,'train_q.json')
        train_sent=read_json(train_path)
        print 'Tokenizing Training Sentences..'
        for i,row in enumerate(train_sent):
            sent=row['question']
            self.tokenize(sent,True)  
        print 'Dictionary Creation done!'
        print 'Dumping dictionary to pkl...'
        self.idx2word.append('UNK')
        self.word2idx['UNK']=len(self.idx2word)-1       
        dump_pkl(self.opt.ANET_DICT_DIR,[self.word2idx,self.idx2word])

    def tgif_dict_token_sent(self):        
        all_sent=self.get_all_sentence()
        print 'Starting to create the dictionary'
        for i,sent in enumerate(all_sent):
            self.tokenize(sent,True)
        print 'Dictionary Creation done!'
        print 'Dumping dictionary to pkl...'
        dump_pkl(self.opt.DICT_DIR,[self.word2idx,self.idx2word])
        
    def tvqa_dict_token_sent(self):
        train_path=os.path.join(self.opt.TVQA_TEXT_DIR,'tvqa_train_processed.json')
        val_path=os.path.join(self.opt.TVQA_TEXT_DIR,'tvqa_val_processed.json')
        print 'Tokenizing Training Sentences..'
        train_data=read_json(train_path)
        val_data=read_json(val_path)
        all_sent=[]
        self.word_count={}
        for i,info in enumerate(train_data):
            sent=[info['a1'],info['a2'],info['a3'],info['a4'],info['a0'],info['q']]
            if self.opt.WITH_TS:
                sent.append(info['located_sub_text'])
            else: 
                sent.append(info['sub_text'])
            all_sent.extend(sent)
        for i,info in enumerate(val_data):
            sent=[info['a1'],info['a2'],info['a3'],info['a4'],info['a0'],info['q']]
            if self.opt.USE_SUBTITLE:
                sent.append(info['located_sub_text'])
            all_sent.extend(sent) 
        for i,sent in enumerate(all_sent):
            self.tokenize_tvqa(sent,True)
        count=0
        for i,word in enumerate(self.word_count.keys()):
            if self.word_count[word]>=self.opt.TVQA_MIN_OCC:
                self.idx2word.append(word)
                self.word2idx[word]=count
                count+=1
        print 'Dictionary Creation done!'
        print 'Dumping dictionary to pkl...'
        self.idx2word.append('UNK')
        self.word2idx['UNK']=len(self.idx2word)-1  
        dump_pkl(self.opt.TVQA_DICT_DIR,[self.word2idx,self.idx2word])        
        
    def get_all_sentence(self):
        '''different dataset, different ways'''
        self.csv_total_path=os.path.join(self.opt.TEXT_DIR,'Total'+self.CSV_TYPE[self.opt.QUES_TYPE])
        all_sent=[]
        csv_pd=read_csv(self.csv_total_path)
        for row in csv_pd.iterrows():
            if self.opt.QUES_TYPE == 'FrameQA':
                column = ['description','question']
            elif self.opt.QUES_TYPE == 'Count':
                column = ['question']
            else:
                column = ['question', 'a1', 'a2', 'a3', 'a4', 'a5']
            sent = [row[1][col] for col in column if not pd.isnull(row[1][col])]
            all_sent.extend(sent)
        return all_sent
        
    def create_embedding(self):
        word2emb={}
        with open(self.opt.GLOVE_PATH,'r') as f:
            entries=f.readlines()
        emb_dim=len(entries[0].split(' '))-1
        weights=np.zeros((len(self.idx2word),emb_dim),dtype=np.float32)
        for entry in entries:
            word=entry.split(' ')[0]
            word2emb[word]=np.array(map(float,entry.split(' ')[1:]))
        for idx,word in enumerate(self.idx2word):
            if word not in word2emb:
                continue
            weights[idx]=word2emb[word]
        if self.opt.DATASET=='tgif-qa':
            np.save(self.opt.EMB_DIR,weights)
        elif self.opt.DATASET=='tvqa':
            np.save(self.opt.TVQA_EMB_DIR,weights)
        elif self.opt.DATASET=='anet-qa':
            np.save(self.opt.ANET_EMB_DIR,weights)
        return weights
        
    def create_dictionary(self):
        self.word2idx={}
        self.idx2word=[]
        if self.opt.DATASET=='anet-qa':
            self.dict_token_sent()
        elif self.opt.DATASET=='tgif-qa':
            self.tgif_dict_token_sent()
        elif self.opt.DATASET=='tvqa':
            self.tvqa_dict_token_sent()
    
    def init_dict(self):
        if self.opt.CREATE_DICT:
            print 'Creating Dictionary...'
            self.create_dictionary()
        else:
            print 'Loading Dictionary...'
            if self.opt.DATASET=='tgif-qa':
                utils.assert_exits(self.opt.DICT_DIR)
                created_dict=load_pkl(self.opt.DICT_DIR)
            elif self.opt.DATASET=='tvqa':
                utils.assert_exits(self.opt.TVQA_DICT_DIR)
                created_dict=load_pkl(self.opt.TVQA_DICT_DIR)
            elif self.opt.DATASET=='anet-qa':
                utils.assert_exits(self.opt.ANET_DICT_DIR)
                created_dict=load_pkl(self.opt.ANET_DICT_DIR)                
            self.word2idx=created_dict[0]
            self.idx2word=created_dict[1]
        '''This is in need of attention!!!'''
        if (self.opt.CREATE_ANS and self.opt.QUES_TYPE=='FrameQA') or (self.opt.CREATE_ANS and self.opt.DATASET=='anet-qa'):
            print 'Creating Answer Labels...'
            self.create_ans2label()
        if self.opt.CREATE_EMB:
            print 'Creating Embedding...'
            self.glove_weights=self.create_embedding()
        else:
            print 'Loading Embedding...'
            if self.opt.DATASET=='tgif-qa':
                self.glove_weights=np.load(self.opt.EMB_DIR)
            elif self.opt.DATASET=='tvqa':
                self.glove_weights=np.load(self.opt.TVQA_EMB_DIR)
            elif self.opt.DATASET=='anet-qa':
                self.glove_weights=np.load(self.opt.ANET_EMB_DIR)
        self.ntokens()
    
    def ntokens(self):
        self.ntoken=len(self.word2idx)
        print 'Num tokens:',len(self.word2idx)
        return len(self.word2idx)
    
    def __len__(self):
        return len(self.idx2word)
    
    
class Activity_Op(Base_Op):
    def __init__(self,opt,dictionary,mode):
        super(Activity_Op,self).__init__()
        self.opt=opt
        self.dictionary=dictionary
        self.mode=mode
        print 'Loading Answer Info...'
        self.ans2label=load_pkl(self.opt.ANET_ANS2LABEL)
        self.num_ans=len(self.ans2label)
        self.video_dict=read_hdf5(self.opt.ANET_VIDEO_DIR)
        self.entries=self.load_tr_val_entries(mode)
        self.tokenize()
        self.tensorize()
             
        
    def load_tr_val_entries(self,mode):
        print 'Loading Dataset for ',mode
        self.miss=0
        '''can use assert to exclude errors'''
        ques=mode+'_q.json'
        ans=mode+'_a.json'
        ques_path=os.path.join(self.opt.ANET_TEXT_DIR,ques)
        ans_path=os.path.join(self.opt.ANET_TEXT_DIR,ans)
        ques_info=read_json(ques_path)
        ans_info=read_json(ans_path)
        entries=[]
        for i,info in enumerate(ques_info):
            question=info['question']
            answer=ans_info[i]['answer']
            q_id=ans_info[i]['question_id']
            video='v_'+info['video_name']
            ques_id=info['question_id']
            type_q=ans_info[i]['type']
            '''If checked, can be omitted'''
            entry={
                'question':question,
                'answer':answer,
                'question_id':ques_id,
                'video_name':video,
                'question_type':type_q
            }
            if 'v_'+info['video_name'] not in self.opt.MISS_DICT:
                entries.append(entry)
            '''if video in self.video_dict.keys():
                entries.append(entry)
            else:
                print q_id'''
                
        print 'Total miss video is:',self.miss
        return entries
    
    def tokenize(self):
        print('Tokenize Questions...')
        for entry in tqdm(self.entries):
            tokens=self.dictionary.tokenize(entry['question'],False)
            tokens=tokens[:self.opt.SENTENCE_LEN]
            if len(tokens)<self.opt.SENTENCE_LEN:
                padding=[self.dictionary.ntoken]*(self.opt.SENTENCE_LEN-len(tokens))
                tokens=padding+tokens
            entry['q_tokens']=np.array((tokens),dtype=np.int64)
            
    def padding_video(self,video_feature):
        padding_feat=np.zeros([self.opt.ANET_VIDEO_LEN,video_feature.shape[1]])
        num_padding=self.opt.ANET_VIDEO_LEN-video_feature.shape[0]
        if num_padding==0:
            padding_feat=video_feature
        elif num_padding<0:
            steps=np.linspace(0, video_feature.shape[0], num=self.opt.ANET_VIDEO_LEN, endpoint=False, dtype=np.int32)
            padding_feat=video_feature[steps]
        else:
            padding_feat[:-num_padding]=video_feature
        return padding_feat
    
    def tensorize(self):
        print 'Loading Video Features...'
        feat_h5py=read_hdf5(self.opt.ANET_VIDEO_DIR)
        print 'Tesnsorize all Information...'
        for entry in tqdm(self.entries):
            entry['q_tokens']=torch.from_numpy(entry['q_tokens'])
            '''The name of video keys may be modified'''
            video_name=entry['video_name']
            feat=np.array(feat_h5py[video_name],dtype=np.float64)
            tensored_feat=torch.from_numpy(self.padding_video(feat))
            entry['feat']=tensored_feat
            answer=entry['answer']
            target=torch.from_numpy(np.zeros((self.opt.VOC_SIZE),dtype=np.float32))
            if answer not in self.ans2label:
                label=self.ans2label['UNK']
            else:
                label=self.ans2label[answer]
                target[label]=1.0
            entry['label']=target
            
    def __getitem__(self,index):
        entry=self.entries[index]
        ''''Operation Above'''
        feat=entry['feat']
        question=entry['question']
        label=entry['label']
        ''''Operation Above'''
        q_token=entry['q_tokens']
        v_name=entry['video_name']
        q_id=entry['question_id']
        q_type=entry['question_type']
        return feat,q_token,label,q_id
        
    def __len__(self):
        return len(self.entries)
        

        
class TGIF_Op(Base_Op):
    def __init__(self,opt,dictionary,mode):
        super(TGIF_Op,self).__init__()
        self.opt=opt
        self.mode=mode
        self.dictionary=dictionary
        if self.opt.QUES_TYPE=='FrameQA':
            print 'Loading Answer Info...'
            self.ans2label=load_pkl(self.opt.ANS2LABEL)
            self.num_ans=len(self.ans2label)
        self.entries=self.load_tr_val_entries(mode)[:8]
        self.tokenize()
        self.tensorize()
            
    def generate_csv_path(self):
        self.csv_path=os.path.join(self.opt.TEXT_DIR,self.mode+self.dictionary.CSV_TYPE[self.opt.QUES_TYPE])
    
    def load_tr_val_entries(self,mode):
        print 'Loading Dataset for ',mode
        '''can use assert to exclude errors'''
        self.generate_csv_path()
        csv_pd=read_csv(self.csv_path)
        csv_pd=csv_pd.set_index('vid_id')
        idx=list(csv_pd.index)
        entries=[]
        '''whether it can be used in this way?'''
        for i,row in enumerate(csv_pd.iterrows()):
            question=row[1]['question']
            answer=row[1]['answer']
            gif_name=row[1]['gif_name']
            vid=row[1]['key']
            index=idx[i]
            entry={
                'question':question,
                'answer':answer,
                'gif_name':gif_name,
                'index':index,
                'vid_id':vid
            }
            if self.opt.QUES_TYPE=='Trans' or self.opt.QUES_TYPE=='Action':
                a1=row[1]['a1']
                a2=row[1]['a2']
                a3=row[1]['a3']
                a4=row[1]['a4']
                a5=row[1]['a5']
                entry['a1']= a1
                entry['a2']= a2
                entry['a3']= a3
                entry['a4']= a4
                entry['a5']= a5
            entries.append(entry)
        return entries
            
    def padding_choice(self,a):
        if len(a)>self.opt.ANS_LEN:
            a=a[:self.opt.ANS_LEN]
        else:
            padding=[self.dictionary.ntoken]*(self.opt.ANS_LEN-len(a))
            a=padding+a
        return a
        
    def tokenize(self):
        print('Tokenize Questions...')
        for entry in tqdm(self.entries):
            tokens=self.dictionary.tokenize(entry['question'],False)
            tokens=tokens[:self.opt.SENTENCE_LEN]
            if len(tokens)<self.opt.SENTENCE_LEN:
                padding=[self.dictionary.ntoken]*(self.opt.SENTENCE_LEN-len(tokens))
                tokens=padding+tokens
            entry['q_tokens']=np.array((tokens),dtype=np.int64)
            if self.opt.QUES_TYPE=='Trans' or self.opt.QUES_TYPE=='Action':
                a1=self.padding_choice(self.dictionary.tokenize(entry['a1'],False))
                a2=self.padding_choice(self.dictionary.tokenize(entry['a2'],False))
                a3=self.padding_choice(self.dictionary.tokenize(entry['a3'],False))
                a4=self.padding_choice(self.dictionary.tokenize(entry['a4'],False))
                a5=self.padding_choice(self.dictionary.tokenize(entry['a5'],False))
                entry['a1_t']=np.array((a1),dtype=np.int64)
                entry['a2_t']=np.array((a2),dtype=np.int64)
                entry['a3_t']=np.array((a3),dtype=np.int64)
                entry['a4_t']=np.array((a4),dtype=np.int64)
                entry['a5_t']=np.array((a5),dtype=np.int64)
                
    def padding_video(self,video_feature):
        padding_feat=np.zeros([self.opt.VIDEO_LEN,video_feature.shape[1]])
        num_padding=self.opt.VIDEO_LEN-video_feature.shape[0]
        if num_padding==0:
            padding_feat=video_feature
        elif num_padding<0:
            steps=np.linspace(0, video_feature.shape[0], num=self.opt.VIDEO_LEN, endpoint=False, dtype=np.int32)
            padding_feat=video_feature[steps]
        else:
            padding_feat[:-num_padding]=video_feature
        return padding_feat                
            
    def tensorize(self):
        print 'Loading Video Features...'
        feat_h5py=read_hdf5(self.opt.VIDEO_DIR)
        print 'Tesnsorize all Information...'
        for entry in tqdm(self.entries):
            entry['q_tokens']=torch.from_numpy(entry['q_tokens'])
            feat=np.array(feat_h5py[str(entry["vid_id"])],dtype=np.float64)
            tensored_feat=torch.from_numpy(self.padding_video(feat))
            entry['feat']=tensored_feat
            answer=entry['answer']
            '''Here!!!'''
            if self.opt.QUES_TYPE=='FrameQA':
                if answer not in self.ans2label:
                    label=self.ans2label['UNK']
                else:
                    label=self.ans2label[answer]
                target=torch.from_numpy(np.zeros((self.num_ans),dtype=np.float32))
                target[label]=1.0   
            elif self.opt.QUES_TYPE=='Count':
                target=torch.clamp(torch.tensor(answer),1,10).float()
            else:
                target=torch.from_numpy(np.zeros((5),dtype=np.float32))
                target[answer]=1.0
            entry['label']=target                  
            
    def __getitem__(self,index):
        '''this index is for the identity of questions'''
        entry=self.entries[index]   
        feat=entry['feat']
        question=entry['question']
        label=entry['label']      
        q_token=entry['q_tokens']
        gif_name=entry['gif_name']
        ques_id=entry['index']
        if self.opt.QUES_TYPE=='FrameQA' or self.opt.QUES_TYPE=='Count':
            return feat,q_token,label,ques_id
        else:
            a1=torch.from_numpy(entry['a1_t'])
            a2=torch.from_numpy(entry['a2_t'])
            a3=torch.from_numpy(entry['a3_t'])
            a4=torch.from_numpy(entry['a4_t'])
            a5=torch.from_numpy(entry['a5_t'])
            '''print entry['question'],q_token
            print entry['a1'],a1'''
            return feat,q_token,label,ques_id,a1,a2,a3,a4,a5
        
        
    def __len__(self):
        return len(self.entries)
    
class TVQA_Op(Base_Op):
    def __init__(self,opt,dictionary,mode):
        super(TVQA_Op,self).__init__()    
        self.opt=opt
        self.mode=mode
        self.dictionary=dictionary
        self.file_path={
            'test':'tvqa_test_public_processed.json',
            'train':'tvqa_train_processed.json',
            'val':'tvqa_val_processed.json'
        }
        
        self.subtitle=read_json(self.opt.TVQA_SUBTITLE)['sub_text']    
        
        self.entries=self.load_entries(mode)
        self.tokenize()
        self.tensorize()
        
    def load_entries(self,mode):
        full_path=os.path.join(self.opt.TVQA_TEXT_DIR,self.file_path[mode])
        '''Because the dataset is so large load part of it first'''
        t_data=read_json(full_path)
        
        entries=[]
        '''whether to use the time stamp?'''
        for i,row in enumerate(t_data):
            ans_idx=row['answer_idx']
            a0=row['a0']
            a1=row['a1']
            a2=row['a2']
            a3=row['a3']
            a4=row['a4']
            question=row['q']
            vid=row['vid_name']
            sub=row['located_sub_text']
            time_stamp=row['located_frame']
            #print time_stamp
            qid=row['qid']
            entry={
                'question':question,
                'qid':qid,
                'ans_idx':ans_idx,
                'time_stamp':time_stamp,
                'vid':vid,
                'sub_text':sub
            }
            '''the modification is that the question is not padded'''
            '''entry['a0']=question+' '+a0
            entry['a1']=question+' '+a1
            entry['a2']=question+' '+a2
            entry['a3']=question+' '+a3
            entry['a4']=question+' '+a4'''
            entry['a0']=a0
            entry['a1']=a1
            entry['a2']=a2
            entry['a3']=a3
            entry['a4']=a4
            entries.append(entry)
        return entries

    def padding_choice(self,a):
        if len(a)>self.opt.ANS_LEN:
            a=a[:self.opt.ANS_LEN]
        else:
            padding=[self.dictionary.ntoken]*(self.opt.ANS_LEN-len(a))
            a=padding+a
        return a

    def padding_sub(self,sub):
        if len(sub)>self.opt.SUBTITLE_LEN:
            sub=sub[:self.opt.SUBTITLE_LEN]
        else:
            padding=[self.dictionary.ntoken]*(self.opt.SUBTITLE_LEN-len(sub))
            sub=padding+sub
        return sub       
    
    def get_sub(self,vid):
        subtitle=self.subtitle[vid]
        sub=""
        for i,info in enumerate(subtitle):
            mid=info.split(':')
            if len(mid)>1:
                info=info.split(':')[1][1:]+' '
            else:
                info+=info+' '
            sub+=info
        tokens=self.dictionary.tokenize(sub,False)
        tokens=tokens[:self.opt.SUBTITLE_LEN]
        if len(tokens)<self.opt.SUBTITLE_LEN:
            padding=[self.dictionary.ntoken]*(self.opt.SUBTITLE_LEN-len(tokens))
            tokens=padding+tokens
        sub=np.array((tokens),dtype=np.int64)
        return sub
        
    def tokenize(self):
        print('Tokenize Questions...')
        for entry in tqdm(self.entries):
            tokens=self.dictionary.tokenize_tvqa(entry['question'],False)
            tokens=tokens[:self.opt.SENTENCE_LEN]
            if len(tokens)<self.opt.SENTENCE_LEN:
                padding=[self.dictionary.ntoken]*(self.opt.SENTENCE_LEN-len(tokens))
                tokens=padding+tokens
            entry['q_tokens']=np.array((tokens),dtype=np.int64)
            a1=self.padding_choice(self.dictionary.tokenize_tvqa(entry['a0'],False))
            a2=self.padding_choice(self.dictionary.tokenize_tvqa(entry['a1'],False))
            a3=self.padding_choice(self.dictionary.tokenize_tvqa(entry['a2'],False))
            a4=self.padding_choice(self.dictionary.tokenize_tvqa(entry['a3'],False))
            a5=self.padding_choice(self.dictionary.tokenize_tvqa(entry['a4'],False))
            entry['a1_t']=np.array((a1),dtype=np.int64)
            entry['a2_t']=np.array((a2),dtype=np.int64)
            entry['a3_t']=np.array((a3),dtype=np.int64)
            entry['a4_t']=np.array((a4),dtype=np.int64)
            entry['a5_t']=np.array((a5),dtype=np.int64)  
            entry['sub']=np.array((self.padding_sub(self.dictionary.tokenize_tvqa(entry['sub_text'],False))),dtype=np.int64)

    def tensorize(self):
        print 'Loading Video Features...'
        feat_h5py=read_hdf5(self.opt.TVQA_VIDEO_DIR)
        print 'Tesnsorize all Information...'
        for entry in tqdm(self.entries):
            entry['q_tokens']=torch.from_numpy(entry['q_tokens'])
            answer=entry['ans_idx']
            '''Here!!!'''
            target=torch.from_numpy(np.zeros((5),dtype=np.float32))
            target[answer]=1.0
            entry['label']=target                      
    
    def padding_video(self,video_feature):
        padding_feat=np.zeros([self.opt.TVQA_VIDEO_LEN,video_feature.shape[1]])
        num_padding=self.opt.TVQA_VIDEO_LEN-video_feature.shape[0]
        if num_padding==0:
            padding_feat=video_feature
        elif num_padding<0:
            steps=np.linspace(0, video_feature.shape[0], num=self.opt.TVQA_VIDEO_LEN, endpoint=False, dtype=np.int32)
            padding_feat=video_feature[steps]
        else:
            padding_feat[:-num_padding]=video_feature
        return padding_feat           
    
    '''this is modified so the train or val should be changed as well''' 
    def __getitem__(self,index):
        entry=self.entries[index]
        #feat=entry['feat']
        '''modified!'''
        feat=entry['vid']
        question=entry['question']
        label=entry['label']      
        q_token=entry['q_tokens']
        video_name=entry['vid']
        ques_id=entry['qid']
        a1=torch.from_numpy(entry['a1_t'])
        a2=torch.from_numpy(entry['a2_t'])
        a3=torch.from_numpy(entry['a3_t'])
        a4=torch.from_numpy(entry['a4_t'])
        a5=torch.from_numpy(entry['a5_t'])
        s=torch.from_numpy(entry['sub'])
        #print entry['time_stamp'][1],len(entry['time_stamp']),type(entry['time_stamp'])
        loc=torch.randn(2,1)
        loc[0]=entry['time_stamp'][0]
        loc[1]=entry['time_stamp'][1]
        '''print q_token,entry['question']
        print a1,entry['a0']
        print loc'''
        return feat,q_token,label,ques_id,a1,a2,a3,a4,a5,s,loc
        
    def __len__(self):
        return len(self.entries)
    
class TVQA_Iter_Op(Base_Op):
    def __init__(self,opt,dictionary,mode):
        super(TVQA_Iter_Op,self).__init__()    
        self.opt=opt
        self.mode=mode
        self.dictionary=dictionary
        self.file_path={
            'test':'tvqa_test_public.jsonl',
            'train':'tvqa_train.jsonl',
            'val':'tvqa_val.jsonl'
        }
        if self.opt.USE_SUBTITLE:
            self.subtitle=read_json(self.opt.TVQA_SUBTITLE)['sub_text']          
        self.entries=self.load_entries(mode)
        self.feat_h5py=read_hdf5(self.opt.TVQA_VIDEO_DIR)
        self.tokenize()
        #self.tensorize()
        
    def load_entries(self,mode):
        full_path=os.path.join(self.opt.TVQA_TEXT_DIR,self.file_path[mode])
        '''Because the dataset is so large load part of it first'''
        t_data=read_jsonl(full_path)
        
        entries=[]
        '''whether to use the time stamp?'''
        for i,row in enumerate(t_data):
            ans_idx=row['answer_idx']
            a0=row['a0']
            a1=row['a1']
            a2=row['a2']
            a3=row['a3']
            a4=row['a4']
            question=row['q']
            vid=row['vid_name']
            time_stamp=row['ts']
            qid=row['qid']
            entry={
                'question':question,
                'qid':qid,
                'idx':i,
                'ans_idx':ans_idx,
                'time_stamp':time_stamp,
                'vid':vid
            }
            entry['a0']=a0+' '+question
            entry['a1']=a1+' '+question
            entry['a2']=a2+' '+question
            entry['a3']=a3+' '+question
            entry['a4']=a4+' '+question
            entries.append(entry)
        return entries

    def padding_choice(self,a):
        if len(a)>self.opt.ANS_LEN:
            a=a[:self.opt.ANS_LEN]
        else:
            padding=[self.dictionary.ntoken]*(self.opt.ANS_LEN-len(a))
            a=padding+a
        return a

    def get_sub(self,vid):
        subtitle=self.subtitle[vid]
        sub=""
        for i,info in enumerate(subtitle):
            mid=info.split(':')
            if len(mid)>1:
                info=info.split(':')[1][1:]+' '
            else:
                info+=info+' '
            sub+=info
        tokens=self.dictionary.tokenize(sub,False)
        tokens=tokens[:self.opt.SUBTITLE_LEN]
        if len(tokens)<self.opt.SUBTITLE_LEN:
            padding=[self.dictionary.ntoken]*(self.opt.SUBTITLE_LEN-len(tokens))
            tokens=padding+tokens
        sub=np.array((tokens),dtype=np.int64)
        return sub    
    
    def tokenize(self):
        print('Tokenize Questions...')
        for entry in tqdm(self.entries):
            tokens=self.dictionary.tokenize(entry['question'],False)
            tokens=tokens[:self.opt.SENTENCE_LEN]
            if len(tokens)<self.opt.SENTENCE_LEN:
                padding=[self.dictionary.ntoken]*(self.opt.SENTENCE_LEN-len(tokens))
                tokens=padding+tokens
            entry['q_tokens']=np.array((tokens),dtype=np.int64)
            if self.opt.QUES_TYPE=='Trans' or self.opt.QUES_TYPE=='Action':
                a1=self.padding_choice(self.dictionary.tokenize(entry['a0'],False))
                a2=self.padding_choice(self.dictionary.tokenize(entry['a1'],False))
                a3=self.padding_choice(self.dictionary.tokenize(entry['a2'],False))
                a4=self.padding_choice(self.dictionary.tokenize(entry['a3'],False))
                a5=self.padding_choice(self.dictionary.tokenize(entry['a4'],False))
                entry['a1_t']=np.array((a1),dtype=np.int64)
                entry['a2_t']=np.array((a2),dtype=np.int64)
                entry['a3_t']=np.array((a3),dtype=np.int64)
                entry['a4_t']=np.array((a4),dtype=np.int64)
                entry['a5_t']=np.array((a5),dtype=np.int64)  
                if self.opt.USE_SUBTITLE:
                    entry['sub']=self.get_sub(entry['vid'])
                
    def tensorize(self,batch_info):
        print 'Loading Video Features...'
        feat_h5py=read_hdf5(self.opt.TVQA_VIDEO_DIR)
        print 'Tesnsorize all Information...'
        for entry in tqdm(self.entries):
            entry['q_tokens']=torch.from_numpy(entry['q_tokens'])
            feat=np.array(feat_h5py[str(entry["vid"])],dtype=np.float64)
            tensored_feat=torch.from_numpy(self.padding_video(feat))
            entry['feat']=tensored_feat
            answer=entry['ans_idx']
            '''Here!!!'''
            target=torch.from_numpy(np.zeros((5),dtype=np.float32))
            target[answer]=1.0
            entry['label']=target  
                
    def padding_video(self,video_feature):
        padding_feat=np.zeros([self.opt.TVQA_VIDEO_LEN,video_feature.shape[1]])
        num_padding=self.opt.TVQA_VIDEO_LEN-video_feature.shape[0]
        if num_padding==0:
            padding_feat=video_feature
        elif num_padding<0:
            steps=np.linspace(0, video_feature.shape[0], num=self.opt.TVQA_VIDEO_LEN, endpoint=False, dtype=np.int32)
            padding_feat=video_feature[steps]
        else:
            padding_feat[:-num_padding]=video_feature
        return padding_feat                  
    
    def iter_ids(self):
        random.shuffle(self.entries)
        for key in self.entries:
            yield key
    
    def get_batch(self):
        if not hasattr(self,'_batch_it'):
            random.seed(100)
            self._batch_it=itertools.cycle(self.iter_ids())
        batch_feat=np.zeros([self.opt.BATCH_SIZE,self.opt.TVQA_VIDEO_LEN,self.opt.RES_SIZE], dtype=np.float64)
        batch_q_token=np.zeros([self.opt.BATCH_SIZE,self.opt.SENTENCE_LEN],dtype=np.int64)
        batch_a1=np.zeros([self.opt.BATCH_SIZE,self.opt.ANS_LEN],dtype=np.int64)
        batch_a2=np.zeros([self.opt.BATCH_SIZE,self.opt.ANS_LEN],dtype=np.int64)
        batch_a3=np.zeros([self.opt.BATCH_SIZE,self.opt.ANS_LEN],dtype=np.int64)
        batch_a4=np.zeros([self.opt.BATCH_SIZE,self.opt.ANS_LEN],dtype=np.int64)
        batch_a5=np.zeros([self.opt.BATCH_SIZE,self.opt.ANS_LEN],dtype=np.int64)
        batch_sub=np.zeros([self.opt.BATCH_SIZE,self.opt.SUBTITLE_LEN],dtype=np.int64)
        batch_target=np.zeros([self.opt.BATCH_SIZE,5],dtype=np.float32)
        batch_info={}
        for k in range(self.opt.BATCH_SIZE):
            chunck=next(self._batch_it)
            batch_feat[k]=self.padding_video(np.array(self.feat_h5py[str(chunck["vid"])],dtype=np.float64))
            batch_q_token[k]=chunck['q_tokens']
            #print chunck['qid']
            batch_a1[k]=chunck['a1_t']
            batch_a2[k]=chunck['a2_t']
            batch_a3[k]=chunck['a3_t']
            batch_a4[k]=chunck['a4_t']
            batch_a5[k]=chunck['a5_t']
            answer=chunck['ans_idx']
            batch_target[k,answer]=1.0
            if self.opt.USE_SUBTITLE:
                batch_sub[k]=chunck['sub']
        batch_info['feat']=torch.from_numpy(batch_feat) 
        batch_info['q_token']=torch.from_numpy(batch_q_token)
        batch_info['a1']=torch.from_numpy(batch_a1)
        batch_info['a2']=torch.from_numpy(batch_a2)
        batch_info['a3']=torch.from_numpy(batch_a3)
        batch_info['a4']=torch.from_numpy(batch_a4)
        batch_info['a5']=torch.from_numpy(batch_a5)
        batch_info['label']=torch.from_numpy(batch_target)
        if self.opt.USE_SUBTITLE:
            batch_info['sub']=torch.from_numpy(batch_sub)        
        while True:
            yield batch_info
    
    def batch_iter(self,batch):
        return batch.next()
    
