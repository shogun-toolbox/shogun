from numpy import mat, transpose
from numpy import array,arange,double

from shogun.Features import StringCharFeatures,DNA
from shogun.Kernel import WeightedDegreeStringKernel

# create toy data

degree=20;
seqlen=60;

strings=[seqlen*'A', seqlen*'C', 'ACTGAAGAAGATCTGAATAAATTTGAGTCTCTTACCATGGGGGCAAAGAAGAAGCTCAAG']

trainfeat = StringCharFeatures(DNA)
trainfeat.set_string_features(strings)

wdk = WeightedDegreeStringKernel(trainfeat, trainfeat, degree, 0)
K = mat(wdk.get_kernel_matrix())
print K

wdk = WeightedDegreeStringKernel(trainfeat, trainfeat, degree, 0)
weights = arange(1,degree+1,dtype=double)[::-1]/sum(arange(1,degree+1,dtype=double))
wdk.set_wd_weights(weights)
K = mat(wdk.get_kernel_matrix())
print K


