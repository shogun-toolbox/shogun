from numpy import mat, transpose
from numpy import array,arange,ones,zeros
from numpy import double,int32

from shogun.Features import CharFeatures,StringCharFeatures,DNA
from shogun.Kernel import WeightedDegreePositionStringKernel, WeightedDegreePositionStringKernel, WeightedDegreeStringKernel

# create toy data

degree=20;
seqlen=60;

strings=['ACTGAAGAAGATCTGAATAAATTTGAGTCTCTTACCATGGGGGCAAAGAAGAAGCTCAAG', seqlen*'A', seqlen*'C', seqlen*'T']

stringfeat = StringCharFeatures(DNA)
stringfeat.set_string_features(strings)

wdk = WeightedDegreePositionStringKernel(stringfeat, stringfeat, degree, zeros(seqlen, dtype=int32))
K = mat(wdk.get_kernel_matrix())
print K

wdk = WeightedDegreePositionStringKernel(stringfeat, stringfeat, degree, 20*ones(seqlen, dtype=int32))
K = mat(wdk.get_kernel_matrix())
print K

wdk = WeightedDegreeStringKernel(stringfeat, stringfeat, degree)
K = mat(wdk.get_kernel_matrix())
print K
