from dataset import Dictionary
 
if __name__=='__main__':
	d=Dictionary()
	all_sent=d.get_all_sentence()

	print all_sent[0],all_sent[1]
	token1=d.tokenize(all_sent[0],False)
	token2=d.tokenize(all_sent[1],False)
	print token1,token2
