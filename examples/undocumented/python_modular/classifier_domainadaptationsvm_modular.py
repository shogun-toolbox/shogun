import numpy

from shogun.Features import StringCharFeatures, Labels, DNA
from shogun.Kernel import WeightedDegreeStringKernel
from shogun.Classifier import SVMLight, DomainAdaptationSVM, MSG_DEBUG

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
label_traindna = NArray.to_na([-1.0]*5 + [1.0]*5)

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
label_testdna = NArray.to_na([-1.0]*5 + [1.0]*5)


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
label_traindna2 = NArray.to_na([-1.0]*5 + [1.0]*5)

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
label_testdna2 = NArray.to_na([-1.0]*5 + [1.0]*5)

parameter_list = [[traindna,testdna,label_traindna,label_testdna,traindna2,label_traindna2, 
                       testdna2,label_testdna2,1,3],[traindna,testdna,label_traindna,label_testdna,traindna2,label_traindna2, 
                       testdna2,label_testdna2,2,5]] 

def classifier_domainadaptationsvm_modular(fm_train_dna=traindna, fm_test_dna=testdna, label_train_dna=label_traindna, label_test_dna=label_testdna, fm_train_dna2=traindna2,fm_test_dna2=testdna2, label_train_dna2=label_traindna2, label_test_dna2=label_testdna2, c=1, degree=3)




	feats_train = Modshogun::StringCharFeatures.new(fm_train_dna, Modshogun::DNA)
	feats_test = Modshogun::StringCharFeatures.new(fm_test_dna, Modshogun::DNA)
	kernel = Modshogun::WeightedDegreeStringKernel.new(feats_train, feats_train, degree)
	labels = Modshogun::Labels.new(label_train_dna)
	svm = Modshogun::SVMLight.new(c, kernel, labels)
	svm.train()
	#svm.io.set_loglevel(MSG_DEBUG)
    
	#####################################
		
	#print "obtaining DA SVM from previously trained SVM"

	feats_train2 = Modshogun::StringCharFeatures.new(fm_train_dna, Modshogun::DNA)
	feats_test2 = Modshogun::StringCharFeatures.new(fm_test_dna, Modshogun::DNA)
	kernel2 = Modshogun::WeightedDegreeStringKernel.new(feats_train, feats_train, degree)
	labels2 = Modshogun::Labels.new(label_train_dna)

	# we regularize against the previously obtained solution
	dasvm = Modshogun::DomainAdaptationSVM.new(c, kernel2, labels2, svm, 1.0)
	dasvm.train()

	out = dasvm.apply(feats_test2).get_labels()

	return out #,dasvm TODO

if __name__=='__main__':
	print 'SVMLight'
	classifier_domainadaptationsvm_modular(*parameter_list[0])
