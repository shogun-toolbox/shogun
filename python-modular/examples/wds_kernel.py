from numpy import mat, transpose
from numpy import array,arange,double

from shogun.Features import StringCharFeatures,DNA
from shogun.Kernel import WeightedPositionStringKernel

# create toy data

degree=20;
seqlen=60;

XT=['ACTGAAGAAGATCTGAATAAATTTGAGTCTCTTACCATGGGGGCAAAGAAGAAGCTCAAG', seqlen*'A', seqlen*'C', seqlen*'T']

trainfeat = CharFeatures(XT,DNA)

wdk = WeightedDegreeCharKernel(trainfeat, trainfeat, degree)
K = mat(wdk.get_kernel_matrix())
print K

weights = arange(1,degree+1,dtype=double)[::-1]/sum(arange(1,degree+1,dtype=double))
wdk = WeightedDegreeCharKernel(trainfeat, trainfeat, degree, weights=weights)
K = mat(wdk.get_kernel_matrix())
print K

wdk = WeightedDegreeCharKernel(trainfeat, trainfeat, degree, block_computation=False, use_normalization=False)
K = mat(wdk.get_kernel_matrix())
print K

weights = arange(1,degree+1,dtype=double)[::-1]/sum(arange(1,degree+1,dtype=double))
wdk = WeightedDegreeCharKernel(trainfeat, trainfeat, degree, block_computation=False, use_normalization=False, weights=weights)
K = mat(wdk.get_kernel_matrix())
print K

weights = arange(1,degree+1,dtype=double)[::-1]/sum(arange(1,degree+1,dtype=double))
wdk = WeightedDegreeCharKernel(trainfeat, trainfeat, degree, block_computation=True, use_normalization=False, weights=weights)
K = mat(wdk.get_kernel_matrix())
print K

weights = arange(1,degree+1,dtype=double)
wdk = WeightedDegreeCharKernel(trainfeat, trainfeat, degree, block_computation=False, use_normalization=False, weights=weights)
K = mat(wdk.get_kernel_matrix())
print K

weights = arange(1,degree+1,dtype=double)
wdk = WeightedDegreeCharKernel(trainfeat, trainfeat, degree, block_computation=True, use_normalization=False, weights=weights)
K = mat(wdk.get_kernel_matrix())
print K

