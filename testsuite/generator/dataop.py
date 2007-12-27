"""
Common operations on train and test data
"""

import sys
import md5
from numpy import *
from numpy.random import seed, rand, randint, permutation
from shogun.Features import Labels

NUM_FEATS=11
NUM_VEC_TRAIN=11
NUM_VEC_TEST=17
LEN_SEQ=60

# need a seed which is always the same for the current entity (kernel,
# distance, etc) to be computed, but at least different between modules
# the seed will only change if the module's filename or the name of the
# function in which the entitity is computed will change.
def _get_seed ():
	fcode=sys._getframe(2).f_code
	hash=reduce(lambda x,y:x+y, map(ord, fcode.co_name+fcode.co_filename))
	return hash

def get_rand (dattype=double, num_feats=NUM_FEATS, dim_square=False,
	max_train=sys.maxint, max_test=sys.maxint):
	seed(_get_seed())

	if dim_square:
		num_feats=num_vec_train=num_vec_test=dim_square
	else:
		num_vec_train=NUM_VEC_TRAIN
		num_vec_test=NUM_VEC_TEST

	if dattype==double:
		return {
			'train':rand(num_feats, num_vec_train),
			'test':rand(num_feats, num_vec_test)
		}
	elif dattype==chararray:
		ord_a=ord('A')
		ord_z=ord('Z')
		rand_train=randint(ord_a, chr_z, num_feats*num_vec_train)
		rand_test=randint(ord_a, chr_z, num_feats*num_vec_test)
		return {
			'train': array(map(lambda x: chr(x), rand_train)).reshape(
				num_feats, num_vec_train),
			'test': array(map(lambda x: chr(x), rand_test)).reshape(
				num_feats, num_vec_train)
		}
	else:
		if dattype==ushort:
			maxval=2**16-1
			if max_train>maxval:
				max_train=maxval
			if max_test>maxval:
				max_test=maxval

		# randint does not understand arg dtype
		rand_train=randint(0, max_train, (num_feats, num_vec_train))
		rand_test=randint(0, max_test, (num_feats, num_vec_test))
		return {
			'train':rand_train.astype(dattype),
			'test':rand_test.astype(dattype)
		}

def get_clouds (num_clouds, num_feats=NUM_FEATS):
	seed(_get_seed())
	clouds={}

	data=[rand(num_feats, NUM_VEC_TRAIN)+x/2 for x in xrange(num_clouds)]
	clouds['train']=concatenate(data, axis=1)
	clouds['train']=array([permutation(x) for x in clouds['train']])

	data=[rand(num_feats, NUM_VEC_TEST)+x/2 for x in xrange(num_clouds)]
	clouds['test']=concatenate(data, axis=1)
	clouds['test']=array([permutation(x) for x in clouds['test']])

	return clouds

def get_cubes (num=3):
	leng=50
	rep=5
	weight=1

	# generate a sequence with characters 1-6 drawn from 3 loaded cubes

	# why the heck so complicated in matlab example?
	#a=[]
	#for i in xrange(3):
	#	one=1*ones((1,ceil(leng*rand())))[0]
	#	two=2*ones((1,ceil(leng*rand())))[0]
	#	three=3*ones((1,ceil(leng*rand())))[0]
	#	four=4*ones((1,ceil(leng*rand())))[0]
	#	five=5*ones((1,ceil(leng*rand())))[0]
	#	six=6*ones((1,ceil(leng*rand())))[0]
	#	b=concatenate((one, two, three, four, five, six))
	#	a.append(permutation(len(b))+1)

	#s=[]
	#for i in xrange(len(a[0][1])):
	#	s.append(i*ones(1,ceil(rep*rand())))
	#s=permutation(s)

	#sequence={}
	#for i in xrange(len(s)):
	#	rn=rand();
	#	f(i)=ceil(((1-weight)*rand()+weight)*length(a{s(i)}));
	#	t=randperm(length(a{s(i)}));
	#	r=a{s(i)}(t(1:f(i)));
	#	sequence{1}=[sequence{1} char(r+'0')];
	#end

	sequence=[]
	for i in xrange(num):
		seq=permutation(6)+1
		sequence.append(''.join([str(x) for x in seq]))

	return {'train':sequence, 'test':sequence}

def get_labels (num, ltype='twoclass'):
	seed(_get_seed())
	labels=[]
	if ltype=='twoclass':
		labels.append(rand(num).round()*2-1)
	elif ltype=='series':
		labels.append([double(x) for x in xrange(num)])
	else:
		return [None, None]

	# essential to wrap in array(), will segfault sometimes otherwise
	labels.append(Labels(array(labels[0])))

	return labels

def get_dna (len_seq_test_add=0):
	seed(_get_seed())
	acgt=array(['A', 'C', 'G','T'])
	len_acgt=len(acgt)
	rand_train=[]
	rand_test=[]

	for i in xrange(NUM_VEC_TRAIN):
		str1=[]
		str2=[]
		for j in range(LEN_SEQ):
			str1.append(acgt[floor(len_acgt*rand())])
			str2.append(acgt[floor(len_acgt*rand())])
		rand_train.append(''.join(str1))
	rand_test.append(''.join(str2))
	
	for i in xrange(NUM_VEC_TEST-NUM_VEC_TRAIN):
		str1=[]
		for j in range(LEN_SEQ+len_seq_test_add):
			str1.append(acgt[floor(len_acgt*rand())])
	rand_test.append(''.join(str1))

	return {'train': rand_train, 'test': rand_test}


