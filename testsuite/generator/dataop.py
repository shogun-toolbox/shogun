"""Common operations on train and test data"""

import sys
import md5
from numpy import *
from numpy.random import seed, rand, randint, permutation
from shogun.Features import Labels

NUM_FEATS=11
NUM_VEC_TRAIN=11
NUM_VEC_TEST=17
LEN_SEQ=60

def _get_seed ():
	"""Return a somewhat constant seed.

	Need a seed which is always the same for the current entity (kernel,
	distance, etc) to be computed, but at least different between modules
	the seed will only change if the module's filename or the name of the
	function in which the entitity is computed will change.
	"""

	fcode=sys._getframe(2).f_code
	hash=reduce(lambda x,y:x+y, map(ord, fcode.co_name+fcode.co_filename))
	return hash

def get_rand (dattype=double, num_feats=NUM_FEATS, dim_square=False,
	max_train=sys.maxint, max_test=sys.maxint):
	"""Return random numbers.

	Return random numbers, either float or integer, in a dict with elements
	'train' and 'test'.

	@param dattype (numpy) data type of the random numbers
	@param num_feats number of features for in a train/test vector
	@param dim_square Dimensions of the square of random numbers, implying that number of features and vectors in both train/test are all the same.
	@param max_train Maximum value for data in train vectors
	@param max_test Maximum value for data in test vectors
	@return Dict which contains the random numbers
	"""

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
		rand_train=randint(ord_a, ord_z, num_feats*num_vec_train)
		rand_test=randint(ord_a, ord_z, num_feats*num_vec_test)
		return {
			'train': array(map(lambda x: chr(x), rand_train)).reshape(
				num_feats, num_vec_train),
			'test': array(map(lambda x: chr(x), rand_test)).reshape(
				num_feats, num_vec_test)
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
	"""Return random float numbers organised, but scrambled, in clouds.

	The float numbers generated here are first created in number clouds, shifted from each other by a constant value. Then they are permutated to hide these clusters a bit when used in classifiers or clustering methods.
	It is a specialised case of random float number generation.

	@param num_clouds Number of clouds to generate
	@param num_features Number of features in each cloud
	@return Dict which contains the random numbers
	"""

	seed(_get_seed())
	clouds={}

	data=[rand(num_feats, NUM_VEC_TRAIN)+x/2 for x in xrange(num_clouds)]
	clouds['train']=concatenate(data, axis=1)
	clouds['train']=array([permutation(x) for x in clouds['train']])

	data=[rand(num_feats, NUM_VEC_TEST)+x/2 for x in xrange(num_clouds)]
	clouds['test']=concatenate(data, axis=1)
	clouds['test']=array([permutation(x) for x in clouds['test']])

	return clouds

def get_cubes (num=4):
	"""Return cubes of with random emissions.

	Used by the Hidden-Markov-Model, it creates a seemingly random sequence of emissions of a 6-sided cube.

	@param num Number of hidden cubes
	@return Dict of tuples of emissions, representing a hidden cube
	"""

	leng=50
	rep=5
	weight=1

	sequence=[]
	for i in xrange(num):
		# generate a sequence with characters 1-6 drawn from 3 loaded cubes
		loaded=[]
		for j in xrange(3):
			draw=[x*ones((1, ceil(leng*rand())), int)[0] for x in xrange(1, 7)]
			loaded.append(permutation(concatenate(draw)))

		draws=[]
		for j in xrange(len(loaded)):
			draws=concatenate((j*ones((1, ceil(rep*rand())), int)[0], draws))
		draws=permutation(draws)

		seq=[]
		for j in xrange(len(draws)):
			len_loaded=len(loaded[draws[j]])
			weighted=int(ceil(((1-weight)*rand()+weight)*len_loaded))
			perm=permutation(len_loaded)
			shuffled=[str(loaded[draws[j]][x]) for x in perm[:weighted]]
			seq=concatenate((seq, shuffled))

		sequence.append(''.join(seq))

	return {'train':sequence, 'test':sequence}

def get_labels (num, ltype='twoclass'):
	"""Return labels used for classification.

	@param num Number of labels
	@param ltype Type of labels, either twoclass or series.
	@return Tuple to contain the labels as numbers in a tuple and labels as objects digestable for Shogun.
	"""

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
	"""Return a random DNA sequence.

	@param len_seq_test_add Additional length of the sequence of characters in the test data, used by some distances.
	@return Dict of tuples of DNA sequences.
	"""

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


