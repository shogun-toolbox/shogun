"""
Common operations on train and test data
"""

from sys import maxint
from numpy import *
from numpy.random import *

ROWS=11
LEN_TRAIN=11
LEN_TEST=17
LEN_SEQ=60
LEN_SEQ_TEST_EXTEND=0

def get_rand (dattype=double, rows=ROWS, dim_square=False, max_train=maxint,
	max_test=maxint):
	if dim_square:
		rows=cols_train=cols_test=dim_square
	else:
		cols_train=LEN_TRAIN
		cols_test=LEN_TEST

	if dattype==double:
		return {'train':rand(rows, cols_train), 'test':rand(rows, cols_test)}
	elif dattype==chararray:
		dtrain=randint(65, 90, rows*cols_train)
		dtest=randint(65, 90, rows*cols_test)
		return {
			'train':
				array(map(lambda x: chr(x), dtrain)).reshape(rows, cols_train),
			'test':
				array(map(lambda x: chr(x), dtest)).reshape(rows, cols_test)
		}
	else:
		if dattype==ushort:
			maxval=2**16-1
			if max_train>maxval:
				max_train=maxval
			if max_test>maxval:
				max_test=maxval

		# randint does not understand arg dtype
		dtrain=randint(0, max_train, (rows, cols_train))
		dtest=randint(0, max_test, (rows, cols_test))
		return {'train':dtrain.astype(dattype), 'test':dtest.astype(dattype)}

def get_dna ():
	acgt=array(['A', 'C', 'G','T'])
	len_acgt=len(acgt)
	dtrain=[]
	dtest=[]

	for i in xrange(LEN_TRAIN):
		str1=[]
		str2=[]
		for j in range(LEN_SEQ):
			str1.append(acgt[floor(len_acgt*rand())])
			str2.append(acgt[floor(len_acgt*rand())])
		dtrain.append(''.join(str1))
	dtest.append(''.join(str2))
	
	for i in xrange(LEN_TEST-LEN_TRAIN):
		str1=[]
		for j in range(LEN_SEQ+LEN_SEQ_TEST_EXTEND):
			str1.append(acgt[floor(len_acgt*rand())])
	dtest.append(''.join(str1))

	return {'train': dtrain, 'test': dtest}


