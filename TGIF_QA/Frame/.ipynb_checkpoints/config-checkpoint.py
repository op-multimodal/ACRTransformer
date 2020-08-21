import argparse 

def parse_opt():
	parser=argparse.ArgumentParser()
	parser.add_argument('--QUESTION_TYPE',type=str,default='FrameQA')
	parser.add_argument('--SENTENCE_LEN',type=int,default=16)
	parser.add_argument('--VIDEO_FEAT_DIR',type=str,default='../../data/features')
	parser.add_argument('--TEXT_DIR',type=str,default='../../data/dataset')
	parser.add_argument('--GLOVE_PATH',type=str,default='../../data/glove/glove.6B.300d.txt')
	parser.add_argument('--HDF5_JSON',type=str,default='../../data/result_proposal.json')
	parser.add_argument('--MIN_OCC',type=int,default=1)
	parser.add_argument('--FEATURE_DIR',type=str,default='../../data/features')
    
	parser.add_argument('--NUM_HIDDEN',type=int,default=1024)
	parser.add_argument('--BIDIRECT',type=bool,default=False)
	parser.add_argument('--NUM_LAYER',type=int,default=1)
    
	parser.add_argument('--EMB_DROPOUT',type=float,default=0.0)
	parser.add_argument('--FC_DROPOUT',type=float,default=0.5)
	parser.add_argument('--L_RNN_DROPOUT',type=float,default=0.0)
    
	parser.add_argument('--ANSWER_LEN',type=int,default=16)
	parser.add_argument('--LEARNING_RATE',type=float,default=0.0095)
	parser.add_argument('--MOMENTUM',type=float,default=0.9)    
	parser.add_argument('--topk',type=int,default=35)

	parser.add_argument('--BATCH_SIZE',type=int,default=256)
	parser.add_argument('--NUM_GLIMPSE',type=int,default=2)
	parser.add_argument('--POOLING_SIZE',type=int,default=5)
	parser.add_argument('--C3D_SIZE',type=int,default=1024)
	parser.add_argument('--RES_SIZE',type=int,default=2048)
	parser.add_argument('--VIDEO_LEN',type=int,default=35)

	parser.add_argument('--MID_DIM',type=int,default=500)
	parser.add_argument('--SEED', type=int, default=1111, help='random seed')
	parser.add_argument('--OUTPUT',type=str,default='./save_models')
	parser.add_argument('--EPOCHS',type=int,default=35)
	parser.add_argument('--NUM_PROPOSAL',type=int,default=6)
	parser.add_argument('--GLIMPSE',type=int,default=2)
    
	parser.add_argument('--save',type=int,default=False)
	parser.add_argument('--save_num',type=int,default=0)
	parser.add_argument('--warmup',type=int,default=2000)
	args=parser.parse_args()
	return args
