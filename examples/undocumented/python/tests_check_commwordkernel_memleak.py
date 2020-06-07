#!/usr/bin/env python
import shogun as sg
parameter_list=[[10,7,0,False]]

def tests_check_commwordkernel_memleak (num, order, gap, reverse):
	from shogun import Alphabet,StringCharFeatures,StringWordFeatures,DNA
	from shogun import MSG_DEBUG
	from shogun import CommWordStringKernel

	POS=[num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'TTGT', num*'TTGT', num*'TTGT',num*'TTGT', num*'TTGT',
	num*'TTGT',num*'TTGT', num*'TTGT', num*'TTGT',num*'TTGT', num*'TTGT',
	num*'TTGT',num*'TTGT', num*'TTGT', num*'TTGT',num*'TTGT', num*'TTGT',
	num*'TTGT',num*'TTGT', num*'TTGT', num*'TTGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT']
	NEG=[num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'TTGT', num*'TTGT', num*'TTGT',num*'TTGT', num*'TTGT',
	num*'TTGT',num*'TTGT', num*'TTGT', num*'TTGT',num*'TTGT', num*'TTGT',
	num*'TTGT',num*'TTGT', num*'TTGT', num*'TTGT',num*'TTGT', num*'TTGT',
	num*'TTGT',num*'TTGT', num*'TTGT', num*'TTGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT',num*'ACGT', num*'ACGT',
	num*'ACGT',num*'ACGT', num*'ACGT', num*'ACGT']

	for i in range(10):
		alpha=Alphabet(DNA)
		traindat=StringCharFeatures(alpha)
		traindat.set_features(POS+NEG)
		trainudat=StringWordFeatures(traindat.get_alphabet());
		trainudat.obtain_from_char(traindat, order-1, order, gap, reverse)
		#trainudat.io.set_loglevel(MSG_DEBUG)
		pre = sg.create_transformer("SortWordString")
		#pre.io.set_loglevel(MSG_DEBUG)
		pre.fit(trainudat)
		trainudat = pre.transform(trainudat)
		spec = CommWordStringKernel(10, False)
		spec.set_normalizer(sg.create_kernel_normalizer("IdentityKernelNormalizer"))
		spec.init(trainudat, trainudat)
		K=spec.get_kernel_matrix()

	del POS
	del NEG
	del order
	del gap
	del reverse
	return K

if __name__=='__main__':
	print('Leak Check Comm Word Kernel')
	tests_check_commwordkernel_memleak(*parameter_list[0])
