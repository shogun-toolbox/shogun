from numpy import mat, transpose
from numpy import array,arange,ones,zeros
from numpy import double,int32

from shogun.Features import CharFeatures,StringCharFeatures,DNA,Alphabet
from shogun.Kernel import WeightedDegreePositionStringKernel, WeightedDegreeCharKernel

# create toy data

degree=20;
seqlen=60;

XT=['ACTGAAGAAGATCTGAATAAATTTGAGTCTCTTACCATGGGGGCAAAGAAGAAGCTCAAG', seqlen*'A', seqlen*'C', seqlen*'T']

charXT=array([list(XT[0]), list(XT[1]), list(XT[2]), list(XT[3])]).T

stringfeat = StringCharFeatures(Alphabet(DNA))
stringfeat.set_string_features(XT)

charfeat = CharFeatures(charXT, DNA)

wdk = WeightedDegreePositionStringKernel(stringfeat, stringfeat, degree, zeros(seqlen, dtype=int32))
K = mat(wdk.get_kernel_matrix())
print K

wdk = WeightedDegreePositionStringKernel(stringfeat, stringfeat, degree, 20*ones(seqlen, dtype=int32))
K = mat(wdk.get_kernel_matrix())
print K

wdk = WeightedDegreeCharKernel(charfeat, charfeat, degree)
K = mat(wdk.get_kernel_matrix())
print K
