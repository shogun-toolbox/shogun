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

wdk = WeightedDegreeStringKernel(trainfeat, trainfeat, degree)
K = mat(wdk.get_kernel_matrix())
print K

weights = arange(1,degree+1,dtype=double)[::-1]/sum(arange(1,degree+1,dtype=double))
wdk = WeightedDegreeStringKernel(trainfeat, trainfeat, degree, weights=weights)
K = mat(wdk.get_kernel_matrix())
print K

wdk = WeightedDegreeStringKernel(trainfeat, trainfeat, degree, block_computation=False, use_normalization=False)
K = mat(wdk.get_kernel_matrix())
print K

weights = arange(1,degree+1,dtype=double)[::-1]/sum(arange(1,degree+1,dtype=double))
wdk = WeightedDegreeStringKernel(trainfeat, trainfeat, degree, block_computation=False, use_normalization=False, weights=weights)
K = mat(wdk.get_kernel_matrix())
print K

weights = arange(1,degree+1,dtype=double)[::-1]/sum(arange(1,degree+1,dtype=double))
wdk = WeightedDegreeStringKernel(trainfeat, trainfeat, degree, block_computation=True, use_normalization=False, weights=weights)
K = mat(wdk.get_kernel_matrix())
print K

weights = arange(1,degree+1,dtype=double)
wdk = WeightedDegreeStringKernel(trainfeat, trainfeat, degree, block_computation=False, use_normalization=False, weights=weights)
K = mat(wdk.get_kernel_matrix())
print K

weights = arange(1,degree+1,dtype=double)
wdk = WeightedDegreeStringKernel(trainfeat, trainfeat, degree, block_computation=True, use_normalization=False, weights=weights)
K = mat(wdk.get_kernel_matrix())
print K

