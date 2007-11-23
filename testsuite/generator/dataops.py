from sys import maxint
from numpy import *
from numpy.random import *

ROWS=11
LEN_TRAIN=11
LEN_TEST=17
LEN_SEQ=60

def get_rand (type=double, rows=ROWS, max_train=maxint, max_test=maxint):
	if type==double:
		return {'train':rand(rows, LEN_TRAIN),
			'test':rand(rows, LEN_TEST)}
	else:
		# randint does not understand arg dtype
		train=randint(0, max_train, (rows, LEN_TRAIN))
		test=randint(0, max_test, (rows, LEN_TEST))
		return {'train':train.astype(type), 'test':test.astype(type)}

def get_dna ():
	acgt=array(['A', 'C', 'G','T'])
	len_acgt=len(acgt)
	train=[]
	test=[]

	for i in range(LEN_TRAIN):
		str1=[]
		str2=[]
		for j in range(LEN_SEQ):
			str1.append(acgt[floor(len_acgt*rand())])
			str2.append(acgt[floor(len_acgt*rand())])
		train.append(''.join(str1))
	test.append(''.join(str2))
	
	for i in range(LEN_TEST-LEN_TRAIN):
		str1=[]
		for j in range(LEN_SEQ):
			str1.append(acgt[floor(len_acgt*rand())])
	test.append(''.join(str1))

	return {'train': train, 'test': test}


