import gc
from shogun.Features import Alphabet,StringCharFeatures,StringWordFeatures,DNA
from shogun.PreProc import SortWordString, MSG_DEBUG
from shogun.Kernel import CommWordStringKernel, IdentityKernelNormalizer
from numpy import mat

POS=[100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'TTGT', 100*'TTGT', 100*'TTGT',100*'TTGT', 100*'TTGT', 
100*'TTGT',100*'TTGT', 100*'TTGT', 100*'TTGT',100*'TTGT', 100*'TTGT', 
100*'TTGT',100*'TTGT', 100*'TTGT', 100*'TTGT',100*'TTGT', 100*'TTGT', 
100*'TTGT',100*'TTGT', 100*'TTGT', 100*'TTGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT']
NEG=[100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'TTGT', 100*'TTGT', 100*'TTGT',100*'TTGT', 100*'TTGT', 
100*'TTGT',100*'TTGT', 100*'TTGT', 100*'TTGT',100*'TTGT', 100*'TTGT', 
100*'TTGT',100*'TTGT', 100*'TTGT', 100*'TTGT',100*'TTGT', 100*'TTGT', 
100*'TTGT',100*'TTGT', 100*'TTGT', 100*'TTGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT',100*'ACGT', 100*'ACGT', 
100*'ACGT',100*'ACGT', 100*'ACGT', 100*'ACGT']
order=7
gap=0
reverse=False

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
    trainudat.add_preproc(pre)
    trainudat.apply_preproc()
    spec = CommWordStringKernel(10, False)
    spec.set_normalizer(IdentityKernelNormalizer())
    spec.init(trainudat, trainudat)
    K=mat(spec.get_kernel_matrix())

del POS
del NEG
del order
del gap
del reverse
