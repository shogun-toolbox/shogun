#!/usr/bin/env python
import numpy

from modshogun import StringCharFeatures, BinaryLabels, DNA
from modshogun import WeightedDegreeStringKernel
from modshogun import SVMLight, DomainAdaptationSVM, MSG_DEBUG

traindna = ['CGCACGTACGTAGCTCGAT',
		      'CGACGTAGTCGTAGTCGTA',
		      'CGACGGGGGGGGGGTCGTA',
		      'CGACCTAGTCGTAGTCGTA',
		      'CGACCACAGTTATATAGTA',
		      'CGACGTAGTCGTAGTCGTA',
		      'CGACGTAGTTTTTTTCGTA',
		      'CGACGTAGTCGTAGCCCCA',
		      'CAAAAAAAAAAAAAAAATA',
		      'CGACGGGGGGGGGGGCGTA']
label_traindna = numpy.array(5*[-1.0] + 5*[1.0])
testdna = ['AGCACGTACGTAGCTCGAT',
		      'AGACGTAGTCGTAGTCGTA',
		      'CAACGGGGGGGGGGTCGTA',
		      'CGACCTAGTCGTAGTCGTA',
		      'CGAACACAGTTATATAGTA',
		      'CGACCTAGTCGTAGTCGTA',
		      'CGACGTGGGGTTTTTCGTA',
		      'CGACGTAGTCCCAGCCCCA',
		      'CAAAAAAAAAAAACCAATA',
		      'CGACGGCCGGGGGGGCGTA']
label_testdna = numpy.array(5*[-1.0] + 5*[1.0])


traindna2 = ['AGACAGTCAGTCGATAGCT',
		      'AGCAGTCGTAGTCGTAGTC',
		      'AGCAGGGGGGGGGGTAGTC',
		      'AGCAATCGTAGTCGTAGTC',
		      'AGCAACACGTTCTCTCGTC',
		      'AGCAGTCGTAGTCGTAGTC',
		      'AGCAGTCGTTTTTTTAGTC',
		      'AGCAGTCGTAGTCGAAAAC',
		      'ACCCCCCCCCCCCCCCCTC',
		      'AGCAGGGGGGGGGGGAGTC']
label_traindna2 = numpy.array(5*[-1.0] + 5*[1.0])
testdna2 = ['CGACAGTCAGTCGATAGCT',
		      'CGCAGTCGTAGTCGTAGTC',
		      'ACCAGGGGGGGGGGTAGTC',
		      'AGCAATCGTAGTCGTAGTC',
		      'AGCCACACGTTCTCTCGTC',
		      'AGCAATCGTAGTCGTAGTC',
		      'AGCAGTGGGGTTTTTAGTC',
		      'AGCAGTCGTAAACGAAAAC',
		      'ACCCCCCCCCCCCAACCTC',
		      'AGCAGGAAGGGGGGGAGTC']
label_testdna2 = numpy.array(5*[-1.0] + 5*[1.0])

parameter_list = [[traindna,testdna,label_traindna,label_testdna,traindna2,label_traindna2, \
                       testdna2,label_testdna2,1,3],[traindna,testdna,label_traindna,label_testdna,traindna2,label_traindna2, \
                       testdna2,label_testdna2,2,5]]

def classifier_domainadaptationsvm_modular (fm_train_dna=traindna,fm_test_dna=testdna, \
                                                label_train_dna=label_traindna, \
                                               label_test_dna=label_testdna,fm_train_dna2=traindna2,fm_test_dna2=testdna2, \
                                               label_train_dna2=label_traindna2,label_test_dna2=label_testdna2,C=1,degree=3):




	feats_train = StringCharFeatures(fm_train_dna, DNA)
	feats_test = StringCharFeatures(fm_test_dna, DNA)
	kernel = WeightedDegreeStringKernel(feats_train, feats_train, degree)
	labels = BinaryLabels(label_train_dna)
	svm = SVMLight(C, kernel, labels)
	svm.train()
	#svm.io.set_loglevel(MSG_DEBUG)

	#####################################

	#print("obtaining DA SVM from previously trained SVM")

	feats_train2 = StringCharFeatures(fm_train_dna, DNA)
	feats_test2 = StringCharFeatures(fm_test_dna, DNA)
	kernel2 = WeightedDegreeStringKernel(feats_train, feats_train, degree)
	labels2 = BinaryLabels(label_train_dna)

	# we regularize against the previously obtained solution
	dasvm = DomainAdaptationSVM(C, kernel2, labels2, svm, 1.0)
	dasvm.train()

	out = dasvm.apply_binary(feats_test2)

	return out #,dasvm TODO

if __name__=='__main__':
	print('SVMLight')
	classifier_domainadaptationsvm_modular(*parameter_list[0])
