import os
import pandas as pd
import json
import cPickle as pkl
import numpy as np
import config
import h5py
import torch
from torch.utils.data import Dataset
import utils
import cPickle as pkl
from tqdm import tqdm

CSV_TYPE={'FrameQA': '_frameqa_question.csv',
          'Count': '_count_question.csv',
          'Trans': '_transition_question.csv',
          'Action' : '_action_question.csv'
         }
class Dictionary(object):
    def __init__(self,word2idx=None,idx2word=None):
        self.opt=config.parse_opt()
        self.word2idx=word2idx
        self.idx2word=idx2word

    def create_ans2label(self):
        occurence=self.count_answer(self.opt.MIN_OCC)
        ans2label={}
        label2ans=[]
        label=0
        for answer in occurence:
            label2ans.append(answer)
            ans2label[answer]=label
            label+=1
        label2ans.append('None')
        ans2label['None']=label
        pkl.dump(ans2label,open("./data/ans2label.pkl",'wb'))
        pkl.dump(label2ans,open("./data/label2ans.pkl",'wb'))
        print 'The length of answers is:',len(ans2label)
        return ans2label

    def count_answer(self,min_occurence):
        occurence={}
        all_ans=[]
        trainset,_,_=self.read_from_csv()
        for row in trainset.iterrows():
            ans=row[1]['answer'].replace(',','').split(' ')
            all_ans.extend(ans)
        print 'Manipulating on',len(all_ans),'answers'
        for i,ans in enumerate(all_ans):
            if ans not in occurence:
                occurence[ans]=1
            else:
                occurence[ans]+=1
        for ans in occurence.keys():
            if occurence[ans]<min_occurence:
                occurence.pop(ans)
        print 'The number of answers is',len(occurence)
        return occurence

    def add_word(self,word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word]=len(self.idx2word)-1
        return self.word2idx[word]

    def tokenize(self,sentence,add_word):
        sentence=sentence.lower()
        sentence=sentence.replace(',','').replace('?','').replace('\'s','\'s').replace('.','')
        words=sentence.split()
        tokens=[]
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                tokens.append(self.word2idx[w])
        return tokens

    def read_from_csv(self):
        utils.assert_in_type(self.opt.QUESTION_TYPE)
        train_path=os.path.join(self.opt.TEXT_DIR,('Train'+CSV_TYPE[self.opt.QUESTION_TYPE]))
        test_path=os.path.join(self.opt.TEXT_DIR,('Test'+CSV_TYPE[self.opt.QUESTION_TYPE]))
        utils.assert_exits(train_path)
        utils.assert_exits(test_path)
        text_train=pd.read_csv(train_path,sep='\t')
        text_test=pd.read_csv(test_path,sep='\t')
        text_train=text_train.set_index('vid_id')
        text_test=text_test.set_index('vid_id')
        total_path=os.path.join(self.opt.TEXT_DIR,('Total'+CSV_TYPE[self.opt.QUESTION_TYPE]))
        total_set=pd.read_csv(total_path,sep='\t')
        return text_train,text_test,total_set

    def get_all_sentence(self):
        _,_,total_set=self.read_from_csv()
        all_sent=[]
        if self.opt.QUESTION_TYPE == 'FrameQA':
            column = ['description','question']
        elif self.opt.QUESTION_TYPE == 'Count':
            column = ['question']
        elif self.opt.QUESTION_TYPE == 'Trans':
            column = ['question', 'a1', 'a2', 'a3', 'a4', 'a5']
        elif self.opt.QUESTION_TYPE == 'Action':
            column = ['question', 'a1', 'a2', 'a3', 'a4', 'a5']
        for row in total_set.iterrows():
            sent = [row[1][col] for col in column if not pd.isnull(row[1][col])]
            all_sent.extend(sent)
        return all_sent 

    def create_dictionary(self):
        all_sent=self.get_all_sentence()
        print 'Starting to create the dictionary'
        for i,sent in enumerate(all_sent):
            self.tokenize(sent,True)
        print 'Creation done!'
        print 'The length of the dictionary is:',len(self.idx2word)
        self.dump_to_file()

    def create_embedding(self):
        if os.path.exists('./data/glove_embedding.npy'):
            weights = np.load('./data/glove_embedding.npy')
            return weights
        word2emb={}
        with open(self.opt.GLOVE_PATH,'r') as f:
            entries=f.readlines()
        emb_dim=len(entries[0].split(' '))-1
        print 'Creating glove embedding of dimension',emb_dim
        weights=np.zeros((len(self.idx2word),emb_dim),dtype=np.float32)
        for entry in entries:
            word=entry.split(' ')[0]
            word2emb[word]=np.array(map(float,entry.split(' ')[1:]))
        for idx,word in enumerate(self.idx2word):
            if word not in word2emb:
                continue
            weights[idx]=word2emb[word]
        np.save('./data/glove_embedding.npy',weights)
        return weights

    def init_dict(self):
        self.create_dictionary()
        self.padding_idx=len(self.word2idx)
        self.glove_weights=self.create_embedding()
        if self.opt.QUESTION_TYPE=='FrameQA':
            self.ans2label=self.create_ans2label()

    def dump_to_file(self):
        if not os.path.exists('./data/dictinary.pkl'):
            print 'Dumping dictionary...'
            pkl.dump([self.word2idx,self.idx2word],open('./data/dictinary.pkl','wb'))
        
    def ntokens(self):
        self.ntoken=len(self.word2idx)
        return len(self.word2idx)

    def __len__(self):
        return len(self.idx2word)
    
    
def _read_from_csv():
    opt=config.parse_opt()
    train_path=os.path.join(opt.TEXT_DIR,('Train'+CSV_TYPE[opt.QUESTION_TYPE]))
    test_path=os.path.join(opt.TEXT_DIR,('Test'+CSV_TYPE[opt.QUESTION_TYPE]))
    text_train=pd.read_csv(train_path,sep='\t')
    text_test=pd.read_csv(test_path,sep='\t')
    text_train=text_train.set_index('vid_id')
    text_test=text_test.set_index('vid_id')
    return text_train,text_test

def load_dataset(mode,num=None):
    if mode=='Train':
        textset,_=_read_from_csv()
    else:
        _,textset=_read_from_csv()
    entries=[]
    opt=config.parse_opt()
    hdf5_result=json.load(open(opt.HDF5_JSON,'r'))['results']
    count=0
    idx=list(textset.index)
    if num == None:
        num = len(idx) + 10
    for row in textset.iterrows():
        question=row[1]['question']
        a1=row[1]['question']+row[1]['a1']
        a2=row[1]['question']+row[1]['a2']
        a3=row[1]['question']+row[1]['a3']
        a4=row[1]['question']+row[1]['a4']
        a5=row[1]['question']+row[1]['a5']
        image_idx=idx[count]
        gif_name=row[1]['gif_name']
        answer=row[1]['answer']
        
        if count > num:
            break
        vid=str(row[1]['key'])
        proposal_info=hdf5_result[gif_name[2:]]
        entry={
            'a1':a1,
            'a2':a2,
            'a3':a3,
            'a4':a4,
            'a5':a5,
            'question':question,
            'answer':answer,
            'gif_name':gif_name,
            'index':image_idx,
            'vid_id':vid,
            'proposal_info':proposal_info
            }
        entries.append(entry)
        count+=1
    return entries

class FeatureDataset(Dataset):
    def __init__(self,question_type,dictionary,mode):
        super(FeatureDataset,self).__init__()
        self.opt=config.parse_opt()
        utils.assert_in_type(question_type)
        if question_type=='FrameQA':
            self.ans2label=pkl.load(open('./data/ans2label.pkl','rb'))
            self.label2ans=pkl.load(open('./data/label2ans.pkl','rb'))	
            self.num_ans=len(self.ans2label)
        self.dictionary=dictionary
        entry_path = './data/entries_'+str(mode)+'.pkl'
        print('Load Dataset')
        self.entries = load_dataset(mode)
        print('Dataset\'s length is %d'%(len(self.entries)))
        self.tokenize()
        self.read_from_h5py()
        self.tensorize()
        '''
        if os.path.exists(entry_path):
            print('Load Dataset')
            self.entries=pkl.load(open(entry_path,'rb'))
            print('Dataset\'s length is %d'%(len(self.entries)))
        else :
            print('Load Dataset')
            self.entries = load_dataset(mode)
            print('Dataset\'s length is %d'%(len(self.entries)))
            #len_e=len(self.entries)/self.opt.BATCH_SIZE
            #self.entries=self.entries[0:len_e * self.opt.BATCH_SIZE]
            
            #self.entries=load_dataset(mode)[0:256]
            #self.entries=load_dataset(mode,256)[0:256]
            self.tokenize()
            self.read_from_h5py()
            #self.read_from_json()
            self.tensorize()
            with open(entry_path,'wb') as f:
                pkl.dump(self.entries,f)
       '''

    def read_from_json(self):
        BSN_path = os.path.join(self.opt.FEATURE_DIR,'result_proposal.json')
        with open(BSN_path,'r') as f:
            BSN_data = json.load(f)['results']
        self.BSN = BSN_data
        
    def compute_interval(self,proposal_info):
        start=min(max(int(proposal_info[0]*self.opt.VIDEO_LEN),0),self.opt.VIDEO_LEN)
        end=min(max(int(proposal_info[1]*self.opt.VIDEO_LEN),0),self.opt.VIDEO_LEN)
        return start,end    
    
    def mask_feat(self,o_feat,r_feat,proposal_info):
        conct=torch.cat((o_feat,r_feat),1).float()
        conct_new=conct.repeat(self.opt.NUM_PROPOSAL,1,1)
        mask_tensor=torch.zeros(self.opt.NUM_PROPOSAL,self.opt.VIDEO_LEN)
        for i in range(self.opt.NUM_PROPOSAL):
            start,end=self.compute_interval(proposal_info[i]['frame_segment'])
            mask_tensor[i][start:end]=1.0
        mask_tensor=torch.unsqueeze(mask_tensor,2)
        mask_feat=conct_new * mask_tensor
        mask=torch.squeeze(torch.sum(mask_feat,0))/self.opt.NUM_PROPOSAL +conct
        return mask
        
    def read_from_h5py(self):
        #c3d_path=os.path.join(self.opt.FEATURE_DIR,'C3D_BSN.hdf5')
        #utils.assert_exits(c3d_path)
        
        #res_path=os.path.join(self.opt.FEATURE_DIR,'TGIF_RESNET_pool5.hdf5')
        #utils.assert_exits(res_path)
        #optical_path=os.path.join(self.opt.FEATURE_DIR,'new_action.hdf5')
        #optical=h5py.File(optical_path,'r')
        
        

        #c3d=h5py.File(c3d_path,'r')
        #res=h5py.File(res_path,'r')
        print 'Loading video features...'
        #self.c3d_feat=c3d
        #self.optical=optical
        #self.res_feat=res
        
        total_path=os.path.join(self.opt.FEATURE_DIR,'merge_BSN.hdf5')
        total=h5py.File(total_path,'r')
        self.total=total

    def get_mask(self,name_key,frame_num):
        item = self.BSN[name_key]
        mask_list = []
        if frame_num > self.opt.VIDEO_LEN :
            frame = self.opt.VIDEO_LEN 
        ct4end = self.opt.topk
        ctnow = 0
        for i in item:
            if ctnow >= ct4end:
                break
            ctnow += 1
            mask_vec = np.zeros(frame_num)
            frame_segment =  i['frame_segment']
            start = int(round(frame_segment[0]*frame_num))
            end = int(round(frame_segment[1]*frame_num))
            for num in range(frame_num):
                if num >= start and num <= end:
                    mask_vec[num] = 1
            if frame_num < self.opt.VIDEO_LEN :
                mask_vec = np.pad(mask_vec, (0, self.opt.VIDEO_LEN  - frame_num), 'constant')
            elif frame_num > self.opt.VIDEO_LEN :
                mask_vec = mask_vec[0:35]
            mask_list.append(mask_vec)
        mask_list = np.array(mask_list)
        return mask_list
        
        
    def tokenize(self):
        print('Tokenize')
        for entry in tqdm(self.entries):
            tokens=self.dictionary.tokenize(entry['question'],False)
            a1_t=self.dictionary.tokenize(entry['a1'],False)
            a2_t=self.dictionary.tokenize(entry['a2'],False)
            a3_t=self.dictionary.tokenize(entry['a3'],False)
            a4_t=self.dictionary.tokenize(entry['a4'],False)
            a5_t=self.dictionary.tokenize(entry['a5'],False)
            tokens=tokens[:self.opt.SENTENCE_LEN]
            if len(tokens)<self.opt.SENTENCE_LEN:
                padding=[self.dictionary.padding_idx]*(self.opt.SENTENCE_LEN-len(tokens))
                tokens=padding+tokens
            a1=a1_t[:self.opt.ANSWER_LEN]
            a2=a2_t[:self.opt.ANSWER_LEN]
            a3=a3_t[:self.opt.ANSWER_LEN]
            a4=a4_t[:self.opt.ANSWER_LEN]
            a5=a5_t[:self.opt.ANSWER_LEN] 
            if len(a1)<self.opt.ANSWER_LEN:
                padding=[self.dictionary.padding_idx]*(self.opt.ANSWER_LEN-len(a1))
                a1=padding+a1
            if len(a2)<self.opt.ANSWER_LEN:
                padding=[self.dictionary.padding_idx]*(self.opt.ANSWER_LEN-len(a2))
                a2=padding+a2
            if len(a3)<self.opt.ANSWER_LEN:
                padding=[self.dictionary.padding_idx]*(self.opt.ANSWER_LEN-len(a3))
                a3=padding+a3
            if len(a4)<self.opt.ANSWER_LEN:
                padding=[self.dictionary.padding_idx]*(self.opt.ANSWER_LEN-len(a4))
                a4=padding+a4
            if len(a5)<self.opt.ANSWER_LEN:
                padding=[self.dictionary.padding_idx]*(self.opt.ANSWER_LEN-len(a5))
                a5=padding+a5
            entry['q_tokens']=np.array((tokens),dtype=np.int64)
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
        print('Tesnsorize')
        for entry in tqdm(self.entries):
            question=torch.from_numpy(entry['q_tokens'])
            entry['q_tokens']=question
            answer=entry['answer']
            #c_feat=np.array(self.c3d_feat[entry['vid_id']],dtype=np.float64)
            #c_feat=torch.from_numpy(self.padding_video(c_feat))
            #r_feat=np.array(self.res_feat[entry['vid_id']],dtype=np.float64)
            #r_feat=torch.from_numpy(self.padding_video(r_feat))
            

            #o_feat=np.array(self.optical[entry["vid_id"]],dtype=np.float64)
            #o_feat=torch.from_numpy(self.padding_video(o_feat))
            
            total=np.array(self.total[str(entry["vid_id"])],dtype=np.float64)
            total_feat=torch.from_numpy(self.padding_video(total))
            frame_num = total.shape[0]
            
            name_key = entry['gif_name'][2:-1] + entry['gif_name'][-1]
            #mask = torch.from_numpy(self.get_mask(name_key,frame_num))
            
            #entry['mask']=mask
            #entry['c_feat']=o_feat
            #entry['r_feat']=r_feat
            entry['total_feat']=total_feat
            answer=entry['answer']
            target=torch.from_numpy(np.zeros((5),dtype=np.float32))
            target[answer]=1.0
            entry['label']=target

    def __getitem__(self,index):
        entry=self.entries[index]
#        c_feat=entry['c_feat']
#        r_feat=entry['r_feat']
        total_feat=entry['total_feat']
        question=entry['q_tokens']
        label=entry['label']
        #mask=entry['mask']
        a1=torch.from_numpy(entry['a1_t'])
        a2=torch.from_numpy(entry['a2_t'])
        a3=torch.from_numpy(entry['a3_t'])
        a4=torch.from_numpy(entry['a4_t'])
        a5=torch.from_numpy(entry['a5_t'])
        #print question,a1
        q=entry['question']
        gif_name=entry['gif_name']
        image_index=entry['index']
        return total_feat,question,label,a1,a2,a3,a4,a5,q,gif_name,image_index,entry['a1'],entry['a2'],entry['a3'],entry['a4'],entry['a5']

    def __len__(self):
        return len(self.entries)
