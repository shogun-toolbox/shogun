# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'
parameter_list=[[10,7,0,0]]

def tests_check_commwordkernel_memleak_modular(num, order, gap, reverse)
	import gc

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

	for i in xrange(10):
		alpha=Alphabet(DNA)
		traindat=StringCharFeatures(alpha)
		traindat.set_features(POS+NEG)
		trainudat=StringWordFeatures(traindat.get_alphabet());
		trainudat.obtain_from_char(traindat, order-1, order, gap, reverse)
		#trainudat.io.set_loglevel(MSG_DEBUG)
		pre = SortWordString()
		#pre.io.set_loglevel(MSG_DEBUG)
		pre.init(trainudat)
		trainudat.add_preprocessor(pre)
		trainudat.apply_preprocessor()
		spec = CommWordStringKernel(10, False)
		spec.set_normalizer(IdentityKernelNormalizer())
		spec.init(trainudat, trainudat)
		K=spec.get_kernel_matrix()

	del POS
	del NEG
	del order
	del gap
	del reverse
	return K


end
if __FILE__ == $0
	print 'Leak Check Comm Word Kernel'
	tests_check_commwordkernel_memleak_modular(*parameter_list[0])

end
