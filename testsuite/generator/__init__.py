"""generator

A package to generate testcases for the shogun toolbox.

"""

__license__='GPL v2'
__status__='alpha'
__version__='20071101'
__url__='http://shogun-toolbox.org'

from numpy.random import *
from shogun.Features import *
from kernels import *

dir_output='data/'
len_train=11
len_test =17
len_seq=60

def _get_matrix(km, mat_name='km'):
	line=list()
	lis=list()

	try:
		for x in range(km.shape[0]):
			for y in range(km.shape[1]):
				if(isinstance(km[x,y],(int, long, float, complex))):
					line.append('%.9g' %km[x,y])
				else:
					line.append("'%s'" %km[x,y])
			lis.append(', '.join(line))
			line=list()
	except IndexError:
		for x in range(km.shape[0]):
			if(isinstance(km[x],(int, long, float, complex))):
				line.append('%.9g' %km[x])
			else:
				line.append("'%s'" %km[x])
		lis.append(', '.join(line))
		line=list()
	kmstr=';'.join(lis)
	kmstr=''.join([mat_name, ' = [',kmstr, ']'])

	return kmstr.replace('\n', '')

def _write (output):
	if output is False:
		return

	print 'Writing for kernel:', output[0]

	value_list = output[3].values()
	value_str  = '_'.join([str(x) for x in value_list])
	value_str  = value_str.replace('.', '')

	mfile = open(dir_output+output[0]+'Kernel_'+value_str+'.m', mode='w')
	mfile.write("functionname = '"+output[1]+"'\n")
	mfile.write("kernelname = '"+output[0]+"Kernel'\n")

	for key in output[2].keys():
		mfile.write("%s\n" % _get_matrix(output[2][key], mat_name=key))

	for key in output[3].keys():
		mfile.write(key+' = %r\n' % output[3][key])

	mfile.close()

def _get_data_dna ():
	acgt=array(['A', 'C', 'G','T'])
	len_acgt=len(acgt)
	train=[]
	test=[]

	for i in range(len_train):
		str1=[]
		str2=[]
		for j in range(len_seq):
			str1.append(acgt[floor(len_acgt*rand())])
			str2.append(acgt[floor(len_acgt*rand())])
		train.append(''.join(str1))
	test.append(''.join(str2))
	
	for i in range(len_test-len_train):
		str1=[]
		for j in range(len_seq):
			str1.append(acgt[floor(len_acgt*rand())])
	test.append(''.join(str1))

	return {'train': train, 'test': test}

def _run_realfeats (data={}, feats={}):
	rows=11
	data['train']=rand(rows, len_train)
	data['test']=rand(rows, len_test)
	feats['train']=RealFeatures(data['train'])
	feats['test']=RealFeatures(data['test'])

	_write(gaussian(feats, data))
	_write(linear(feats, data))
	_write(chi2(feats, data))
	_write(sigmoid(feats, data, 1.1, 1.3))
	_write(sigmoid(feats, data, 0.5, 0.7))
	_write(poly(feats, data, True, True))
	_write(poly(feats, data, False, True))
	_write(poly(feats, data, True, False))
	_write(poly(feats, data, False, False))
	_write(svm_gaussian(feats, data, 1.5))

def _run_stringfeats (data={}, feats={}):
	data = _get_data_dna()
	feats['train']=StringCharFeatures(DNA)
	feats['train'].set_string_features(data['train'])
	feats['test']=StringCharFeatures(DNA)
	feats['test'].set_string_features(data['test'])

	_write(weighted_degree_string(feats, data, len_seq))
	_write(weighted_degree_position_string(feats, data, len_seq))
	_write(common_word_string(feats, data, len_seq))

def run ():
	# ASK: really necessary to seed explicitely?
	#seed(None)
	seed(42)

	_run_realfeats()
	_run_stringfeats()


