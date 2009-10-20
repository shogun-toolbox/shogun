from shogun.Features import *
from shogun.Features import StringCharFeatures, Labels, DNA, Alphabet
from shogun.Kernel import WeightedDegreeStringKernel
from shogun.Classifier import SVMLight
import numpy


degree=3
fm_train_dna=['CGCACGTACGTAGCTCGAT',
		      'CGACGTAGTCGTAGTCGTA',
		      'CGACGGGGGGGGGGTCGTA',
		      'CGACCTAGTCGTAGTCGTA',
		      'CGACCACAGTTATATAGTA',
		      'CGACGTAGTCGTAGTCGTA',
		      'CGACGTAGTTTTTTTCGTA',
		      'CGACGTAGTCGTAGCCCCA',
		      'CAAAAAAAAAAAAAAAATA',
		      'CGACGGGGGGGGGGGCGTA']
label_train_dna=numpy.array(5*[-1.0] + 5*[1.0])
fm_test_dna=['AGCACGTACGTAGCTCGAT',
		      'AGACGTAGTCGTAGTCGTA',
		      'CAACGGGGGGGGGGTCGTA',
		      'CGACCTAGTCGTAGTCGTA',
		      'CGAACACAGTTATATAGTA',
		      'CGACCTAGTCGTAGTCGTA',
		      'CGACGTGGGGTTTTTCGTA',
		      'CGACGTAGTCCCAGCCCCA',
		      'CAAAAAAAAAAAACCAATA',
		      'CGACGGCCGGGGGGGCGTA']
label_test_dna=numpy.array(5*[-1.0] + 5*[1.0])


##################################################

alpha = Alphabet(DNA)

feats_train=StringCharFeatures(DNA)
feats_train.set_features(fm_train_dna)
feats_test=StringCharFeatures(DNA)
feats_test.set_features(fm_test_dna)

kernel=WeightedDegreeStringKernel(feats_train, feats_train, degree)

C=10
epsilon=1e-5
num_threads=1
labels=Labels(label_train_dna)

svm=SVMLight(C, kernel, labels)
svm.set_qpsize(3)
svm.set_linear_term(-numpy.array([1,2,3,4,5,6,7,8,7,6], dtype=numpy.double));
svm.set_epsilon(epsilon)
svm.parallel.set_num_threads(num_threads)
svm.train()

##################################################

print "alphabet:"
print alpha.to_string()

print "labels:"
print labels.to_string()

print "features"
print feats_train.to_string()

print "kernel"
print kernel.to_string()

print "svm"
print svm.to_string()



