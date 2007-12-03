from sys import maxint
from numpy import *
from numpy.random import *

ROWS=11
LEN_TRAIN=11
LEN_TEST=17
LEN_SEQ=60

def get_rand (type=double, rows=ROWS, dim_square=False, max_train=maxint, max_test=maxint):
	if dim_square:
		rows=cols_train=cols_test=dim_square
	else:
		cols_train=LEN_TRAIN
		cols_test=LEN_TEST

	if type==double:
		return {'train':rand(rows, cols_train), 'test':rand(rows, cols_test)}
	elif type==chararray:
		a=randint(65, 90, rows*cols_train)
		b=randint(65, 90, rows*cols_test)
		return {
			'train':array(map(lambda x: chr(x), a)).reshape(rows, cols_train),
			'test':array(map(lambda x: chr(x), b)).reshape(rows, cols_test)
		}
	else:
		if type==ushort:
			max=2**16-1
			if max_train>max: max_train=max
			if max_test>max: max_test=max

		# randint does not understand arg dtype
		train=randint(0, max_train, (rows, cols_train))
		test=randint(0, max_test, (rows, cols_test))
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


