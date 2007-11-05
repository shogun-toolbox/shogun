"""generator

A package to generate testcases for the shogun toolbox.

"""

__license__='GPL v2'
__status__='alpha'
__version__='20071101'
__url__='http://shogun-toolbox.org'

from numpy.random import *
from shogun.Features import *
import kernels

DIR_OUTPUT='data/'

def _get_matrix (name, km):
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
	kmstr=''.join([name, ' = [', kmstr, ']'])

	return kmstr.replace('\n', '')

def _get_filename (output):
	params=[]

	for v in output[2].itervalues():
		cn=v.__class__.__name__
		if cn!='ndarray' and cn!='matrix':
			params.append(str(v))

	params='_'.join(params).replace('.', '')
	return DIR_OUTPUT+output[0]+'Kernel_'+params+'.m'

def _write (output):
	if output is False:
		return

	print 'Writing for kernel:', output[0]

	prefix_svm='svm_'
	mfile=open(_get_filename(output), mode='w')
	
	if (output[0].startswith(prefix_svm)):
		output[0]=output[0][len(prefix_svm):] # remove for kernel's name

	# need suffix Kernel b/c matlab et al not as sophisticated as python
	# in string processing
	mfile.write("name = '"+output[0]+"Kernel'\n")

	data=output[1]
	data.update(output[2])
	for k,v in data.iteritems():
		cn=v.__class__.__name__
		if cn=='bool' or cn=='str':
			mfile.write("%s = '%s'\n"%(k, v))
		elif cn=='ndarray' or cn=='matrix':
			mfile.write("%s\n"%_get_matrix(k, v))
		else:
			mfile.write("%s = %s\n"%(k, v))

	mfile.close()


def _run_realfeats ():
	data=kernels.get_data_rand()
	feats=kernels.get_feats_real(data)

	_write(kernels.gaussian(feats, data))
	_write(kernels.linear(feats, data))
	_write(kernels.chi2(feats, data))
	_write(kernels.sigmoid(feats, data, 1.1, 1.3))
	_write(kernels.sigmoid(feats, data, 0.5, 0.7))
	_write(kernels.poly(feats, data, 3, True, True))
	_write(kernels.poly(feats, data, 3, False, True))
	_write(kernels.poly(feats, data, 3, True, False))
	_write(kernels.poly(feats, data, 3, False, False))
	_write(kernels.svm_gaussian(feats, data, 1.5))

def _run_stringfeats ():
	data=kernels.get_data_dna()
	feats=kernels.get_feats_string(data)

	_write(kernels.weighted_degree_string(feats, data))
	_write(kernels.weighted_degree_position_string(feats, data))
	#_write(kernels.locality_improved_string(feats, data))

def _run_wordfeats ():
	data=kernels.get_data_dna()
	feats=kernels.get_feats_word(data)

	_write(kernels.common_word_string(feats, data))
	#_write(kernels.manhattan_word(feats, data))
	#_write(kernels.hamming_word(feats, data, 50, 10, False))

def run ():
	# ASK: really necessary to seed explicitely?
	#seed(None)
	seed(42)

	_run_realfeats()
	_run_stringfeats()
	_run_wordfeats()


