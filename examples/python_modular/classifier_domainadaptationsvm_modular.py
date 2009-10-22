import numpy

degree=3
fm_train_dna = ['CGCACGTACGTAGCTCGAT',
		      'CGACGTAGTCGTAGTCGTA',
		      'CGACGGGGGGGGGGTCGTA',
		      'CGACCTAGTCGTAGTCGTA',
		      'CGACCACAGTTATATAGTA',
		      'CGACGTAGTCGTAGTCGTA',
		      'CGACGTAGTTTTTTTCGTA',
		      'CGACGTAGTCGTAGCCCCA',
		      'CAAAAAAAAAAAAAAAATA',
		      'CGACGGGGGGGGGGGCGTA']
label_train_dna = numpy.array(5*[-1.0] + 5*[1.0])
fm_test_dna = ['AGCACGTACGTAGCTCGAT',
		      'AGACGTAGTCGTAGTCGTA',
		      'CAACGGGGGGGGGGTCGTA',
		      'CGACCTAGTCGTAGTCGTA',
		      'CGAACACAGTTATATAGTA',
		      'CGACCTAGTCGTAGTCGTA',
		      'CGACGTGGGGTTTTTCGTA',
		      'CGACGTAGTCCCAGCCCCA',
		      'CAAAAAAAAAAAACCAATA',
		      'CGACGGCCGGGGGGGCGTA']
label_test_dna = numpy.array(5*[-1.0] + 5*[1.0])


fm_train_dna2 = ['AGACAGTCAGTCGATAGCT',
		      'AGCAGTCGTAGTCGTAGTC',
		      'AGCAGGGGGGGGGGTAGTC',
		      'AGCAATCGTAGTCGTAGTC',
		      'AGCAACACGTTCTCTCGTC',
		      'AGCAGTCGTAGTCGTAGTC',
		      'AGCAGTCGTTTTTTTAGTC',
		      'AGCAGTCGTAGTCGAAAAC',
		      'ACCCCCCCCCCCCCCCCTC',
		      'AGCAGGGGGGGGGGGAGTC']
label_train_dna2 = numpy.array(5*[-1.0] + 5*[1.0])
fm_test_dna2 = ['CGACAGTCAGTCGATAGCT',
		      'CGCAGTCGTAGTCGTAGTC',
		      'ACCAGGGGGGGGGGTAGTC',
		      'AGCAATCGTAGTCGTAGTC',
		      'AGCCACACGTTCTCTCGTC',
		      'AGCAATCGTAGTCGTAGTC',
		      'AGCAGTGGGGTTTTTAGTC',
		      'AGCAGTCGTAAACGAAAAC',
		      'ACCCCCCCCCCCCAACCTC',
		      'AGCAGGAAGGGGGGGAGTC']
label_test_dna2 = numpy.array(5*[-1.0] + 5*[1.0])


print 'SVMLight'

from shogun.Features import StringCharFeatures, Labels, DNA
from shogun.Kernel import WeightedDegreeStringKernel
from shogun.Classifier import SVMLight, DomainAdaptationSVM

C = 1.0

feats_train=StringCharFeatures(DNA)
feats_train.set_features(fm_train_dna)
feats_test=StringCharFeatures(DNA)
feats_test.set_features(fm_test_dna)

kernel=WeightedDegreeStringKernel(feats_train, feats_train, degree)

labels=Labels(label_train_dna)

svm=SVMLight(C, kernel, labels)
svm.train()


#####################################

print "obtaining DA SVM from previously trained SVM"

feats_train2=StringCharFeatures(DNA)
feats_train2.set_features(fm_train_dna)
feats_test2=StringCharFeatures(DNA)
feats_test2.set_features(fm_test_dna)

kernel2=WeightedDegreeStringKernel(feats_train, feats_train, degree)

labels2=Labels(label_train_dna)

# we regularize versus the previously obtained solution
dasvm = DomainAdaptationSVM(C, kernel2, labels2, svm, 1.0)
dasvm.train()

out = dasvm.classify(feats_test2).get_labels()


