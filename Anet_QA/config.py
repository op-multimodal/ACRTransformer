import argparse

def parse_opt():
    parser=argparse.ArgumentParser()
    '''path related'''
    parser.add_argument('--GLOVE_PATH',type=str,default='./data/glove/glove.6B.300d.txt')   
    '''tgif-qa related'''
    parser.add_argument('--TEXT_DIR',type=str,default='/home/caorui/tgif-qa-master/dataset')
    
    #for TGIF, the text path is just the root path: /home/caorui/tgif-qa-master/dataset
    #parser.add_argument('--DICT_DIR',type=str,default='./tgif-qa/dictionary.pkl')
    #parser.add_argument('--ANS2LABEL',type=str,default='./tgif-qa/ans2label.pkl')
    #parser.add_argument('--LABEL2ANS',type=str,default='./tgif-qa/label2ans.pkl')
    #parser.add_argument('--TVQA_SUBTITLE',type=str,default="/home/caorui/video_feature/TVQA/data/srt_data_cache.json")
    #parser.add_argument('--VIDEO_DIR',type=str,default='/home/caorui/video_feature/merge_BSN.hdf5')
    #parser.add_argument('--EMB_DIR',type=str,default='./tgif-qa/glove_embedding.npy')
    #parser.add_argument('--RESULT_PATH',type=str,default='./tgif-qa/result')
    '''TVQA related path'''
    #parser.add_argument('--TVQA_VIDEO_DIR',type=str,default='/home/caorui/TVQA/tvqa_imagenet_pool5_hq.h5')
    #parser.add_argument('--TVQA_TEXT_DIR',type=str,default='/home/caorui/video_feature/TVQA/data')
    
    #the path for qa pairs is not decided yet
    #parser.add_argument('--TVQA_DICT_DIR',type=str,default='./TVQA/dictionary.pkl')
    #parser.add_argument('--TVQA_ANS2LABEL',type=str,default='./TVQA/ans2label.pkl')
    #parser.add_argument('--TVQA_LABEL2ANS',type=str,default='./TVQA/label2ans.pkl')
    #parser.add_argument('--TVQA_EMB_DIR',type=str,default='./TVQA/glove_embedding.npy')
    #parser.add_argument('--TVQA_RESULT_PATH',type=str,default='./TVQA/result')
    #parser.add_argument('--USE_SUBTITLE',type=bool,default=True)
    #parser.add_argument('--WITH_TS',type=bool,default=True)
    #parser.add_argument('--USE_ITER',type=bool,default=False)  
    #parser.add_argument('--SUBTITLE_LEN',type=int,default=50)
    
    #parser.add_argument('--TVQA_VIDEO_LEN',type=int,default=25)
    
    '''Anet-qa related'''
    parser.add_argument('--ANET_VIDEO_DIR',type=str,default='../data/ANET_BSN.hdf5')
    parser.add_argument('--ANET_TEXT_DIR',type=str,default='../data/dataset')
    parser.add_argument('--ANET_VAL_DICT',type=str,default='./activitynet-qa/dataset/test_dict.pkl')
    parser.add_argument('--ANET_TRAIN_DICT',type=str,default='./activitynet-qa/dataset/train_dict.pkl')
    parser.add_argument('--ANET_QA_DICT',type=str,default='../data/qa_pairs.hdf5')
    #the path for qa pairs is not decided yet
    parser.add_argument('--ANET_DICT_DIR',type=str,default='./anet-qa/dictionary.pkl')
    parser.add_argument('--ANET_QA_RESULT',type=str,default='./anet-qa/type_result/type_result')
    parser.add_argument('--ANET_ANS2LABEL',type=str,default='./anet-qa/ans2label.pkl')
    parser.add_argument('--MISS_DICT',type=dict,default={'v_smJtFktW640':0,'v_j73Wh1olDsA':0,'v_mua8hNPuQHw':0,'v_Vhn4SuPhu-0':0})
    parser.add_argument('--ANET_LABEL2ANS',type=str,default='./anet-qa/label2ans.pkl')
    parser.add_argument('--ANET_EMB_DIR',type=str,default='./anet-qa/glove_embedding.npy')
    parser.add_argument('--ANET_RESULT_PATH',type=str,default='./anet-qa/result')
    parser.add_argument('--USE_C3D',type=bool,default=True)
    
    parser.add_argument('--ANET_VIDEO_LEN',type=int,default=40)
    '''basic related'''
    parser.add_argument('--CUDA_DEVICE',type=int,default=1)
    parser.add_argument('--SEED',type=int,default=1111)
    parser.add_argument('--C3D_SIZE',type=int,default=1024)    
    parser.add_argument('--OPTICAL_SIZE',type=int,default=1024)    
    parser.add_argument('--RES_SIZE',type=int,default=2048)    
    parser.add_argument('--DATASET',type=str,default='anet-qa')
    parser.add_argument('--BASIC',type=bool,default=False)
    
    '''training related'''
    parser.add_argument('--EPOCHS',type=int,default=30)
    parser.add_argument('--NUM_ITER',type=int,default=15000)
    parser.add_argument('--NUM_EVALUATE_ITER',type=int,default=1000)
    parser.add_argument('--SAVE_NUM',type=int,default=100)
    
    '''hyper related'''
    parser.add_argument('--EMB_DROPOUT',type=float,default=0.2)
    parser.add_argument('--FC_DROPOUT',type=float,default=0.2) 
    parser.add_argument('--VIDEO_PROJ_DROPOUT',type=float,default=0.3)  
    parser.add_argument('--L_RNN_DROPOUT',type=float,default=0.2)
    parser.add_argument('--ATT_DROPOUT',type=float,default=0.2)  
    parser.add_argument('--ATT_V_DROPOUT',type=float,default=0.2)   
    parser.add_argument('--ATT_Q_DROPOUT',type=float,default=0.2)       
    parser.add_argument('--MIN_OCC',type=int,default=2)
    parser.add_argument('--VOC_SIZE',type=int,default=1000)
    parser.add_argument('--TVQA_MIN_OCC',type=int,default=2)
    
    parser.add_argument('--BATCH_SIZE',type=int,default=128)
    parser.add_argument('--BILINEAR_DIM',type=int,default=1024)
    parser.add_argument('--EMB_DIM',type=int,default=300)
    parser.add_argument('--VIDEO_LEN',type=int,default=35)
    parser.add_argument('--SENTENCE_LEN',type=int,default=13)
    parser.add_argument('--ANS_LEN',type=int,default=8)
    parser.add_argument('--NUM_HIDDEN',type=int,default=1024)
    parser.add_argument('--FINAL_HIDDEN',type=int,default=3072)
    parser.add_argument('--V_HIDDEN',type=int,default=1024) 
    parser.add_argument('--TVQA_HIDDEN',type=int,default=150) 
    parser.add_argument('--MID_DIM',type=int,default=400)
    parser.add_argument('--PROJ_MID_DIM',type=int,default=512)
    parser.add_argument('--RATIO',type=int,default=2)
    parser.add_argument('--NUM_SEG',type=int,default=5)
    parser.add_argument('--NUM_R_SEG',type=int,default=8)
    parser.add_argument('--NUM_O_SEG',type=int,default=5) 
    parser.add_argument('--NUM_LAYER',type=int,default=1)
    parser.add_argument('--GLIMPSE',type=int,default=2)
    parser.add_argument('--BIDIRECT',type=bool,default=False)
    parser.add_argument('--WARM_UP',type=int,default=2000)
    
    
    '''initial related'''
    parser.add_argument('--CREATE_DICT',type=bool,default=False)
    parser.add_argument('--CREATE_EMB',type=bool,default=False)
    parser.add_argument('--CREATE_ANS',type=bool,default=False)
    
    '''TGIF related'''
    '''For tvqa set as Trans or Action'''
    #parser.add_argument('--QUES_TYPE',type=str,default='FrameQA')
    args=parser.parse_args()
    return args
    
