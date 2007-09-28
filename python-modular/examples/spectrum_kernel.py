order=3
gap=0
reverse=False

from shogun.Features import StringCharFeatures,StringWordFeatures,Alphabet,DNA
from shogun.Kernel import CommWordStringKernel
from shogun.PreProc import SortWordString

alpha=Alphabet(DNA)

POS_TESTSET=['ACGTACCATCGATCGAT','CAGATCTACTAGTC'];
NEG_TESTSET=['ACGTAAAAAAAAAAGAT','CAAAAATAGTC'];

traindat=StringCharFeatures(alpha)
traindat.set_string_features(POS_TESTSET+NEG_TESTSET)
trainudat=StringWordFeatures(traindat.get_alphabet());
trainudat.obtain_from_char(traindat, order-1, order, gap, reverse)

pre = SortWordString()
pre.init(trainudat)
trainudat.add_preproc(pre)
trainudat.apply_preproc()

k=CommWordStringKernel(trainudat, trainudat)
km=k.get_kernel_matrix()

print km
