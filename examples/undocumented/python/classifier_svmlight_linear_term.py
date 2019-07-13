#!/usr/bin/env python
import numpy

traindna=['CGCACGTACGTAGCTCGAT',
		      'CGACGTAGTCGTAGTCGTA',
		      'CGACGGGGGGGGGGTCGTA',
		      'CGACCTAGTCGTAGTCGTA',
		      'CGACCACAGTTATATAGTA',
		      'CGACGTAGTCGTAGTCGTA',
		      'CGACGTAGTTTTTTTCGTA',
		      'CGACGTAGTCGTAGCCCCA',
		      'CAAAAAAAAAAAAAAAATA',
		      'CGACGGGGGGGGGGGCGTA']
label_traindna=numpy.array(5*[-1.0] + 5*[1.0])
testdna=['AGCACGTACGTAGCTCGAT',
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

parameter_list = [[traindna,testdna,label_traindna,3,10,1e-5,1],[traindna,testdna,label_traindna,3,10,1e-5,1]]

def classifier_svmlight_linear_term (fm_train_dna=traindna,fm_test_dna=testdna, \
                                                label_train_dna=label_traindna,degree=3, \
                                                C=10,epsilon=1e-5,num_threads=1):

    from shogun import StringCharFeatures, BinaryLabels, DNA
    from shogun import WeightedDegreeStringKernel
    from shogun import machine
    try:
    	machine("SVMLight")
    except SystemError:
    	print("SVMLight is not available")
    	exit(0)

    feats_train=StringCharFeatures(DNA)
    feats_train.set_features(fm_train_dna)
    feats_test=StringCharFeatures(DNA)
    feats_test.set_features(fm_test_dna)

    kernel=WeightedDegreeStringKernel(feats_train, feats_train, degree)

    labels=BinaryLabels(label_train_dna)

    svm=machine("SVMLight", C1=C, C2=C, kernel=kernel, labels=labels, epsilon=epsilon)
    svm.put("qpsize", 3)
    svm.put("linear_term", -numpy.array([1,2,3,4,5,6,7,8,7,6], dtype=numpy.double))
    svm.get_global_parallel().set_num_threads(num_threads)
    svm.train()

    kernel.init(feats_train, feats_test)
    out = svm.apply()
    return out,kernel

if __name__=='__main__':
    print('SVMLight')
    classifier_svmlight_linear_term(*parameter_list[0])
