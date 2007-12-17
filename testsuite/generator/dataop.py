"""
Common operations on train and test data
"""

import sys
import md5
from numpy import double, chararray, ushort, array, floor, concatenate
from numpy.random import seed, rand, randint, permutation

ROWS=11
LEN_TRAIN=11
LEN_TEST=17
LEN_SEQ=60

# need a seed which is always the same for the current entity (kernel,
# distance, etc) to be computed, but at least different between modules
# the seed will only change if the module's filename or the name of the
# function in which the entitity is computed will change.
def _get_seed ():
	fcode=sys._getframe(2).f_code
	hash=reduce(lambda x,y:x+y, map(ord, fcode.co_name+fcode.co_filename))
	return hash

def get_rand (dattype=double, rows=ROWS, dim_square=False,
	max_train=sys.maxint, max_test=sys.maxint):
	seed(_get_seed())

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

def get_clouds (num, rows=ROWS):
	clouds={}
	seed(_get_seed())

	data=[rand(rows, LEN_TRAIN)+x/2 for x in xrange(num)]
	clouds['train']=concatenate(data, axis=0)
	clouds['train']=array([permutation(x) for x in clouds['train']])

	data=[rand(rows, LEN_TEST)+x/2 for x in xrange(num)]
	clouds['test']=concatenate(data, axis=0)
	clouds['test']=array([permutation(x) for x in clouds['test']])

	return clouds

def get_dna (len_seq_test_add=0):
	seed(_get_seed())
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
		for j in range(LEN_SEQ+len_seq_test_add):
			str1.append(acgt[floor(len_acgt*rand())])
	dtest.append(''.join(str1))

	return {'train': dtrain, 'test': dtest}


