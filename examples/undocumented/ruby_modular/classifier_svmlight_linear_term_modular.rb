# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'
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

def classifier_svmlight_linear_term_modular(fm_train_dna=traindna,fm_test_dna=testdna, \
                                                label_train_dna=label_traindna,degree=3, \
                                                C=10,epsilon=1e-5,num_threads=1):
    
    
# ***     feats_train=StringCharFeatures(DNA)
    feats_train=Modshogun::StringCharFeatures.new
    feats_train.set_features(DNA)
    feats_train.set_features(fm_train_dna)
# ***     feats_test=StringCharFeatures(DNA)
    feats_test=Modshogun::StringCharFeatures.new
    feats_test.set_features(DNA)
    feats_test.set_features(fm_test_dna)
    
# ***     kernel=WeightedDegreeStringKernel(feats_train, feats_train, degree)
    kernel=Modshogun::WeightedDegreeStringKernel.new
    kernel.set_features(feats_train, feats_train, degree)
    
# ***     labels=Labels(label_train_dna)
    labels=Modshogun::Labels.new
    labels.set_features(label_train_dna)
    
# ***     svm=SVMLight(C, kernel, labels)
    svm=Modshogun::SVMLight.new
    svm.set_features(C, kernel, labels)
    svm.set_qpsize(3)
    svm.set_linear_term(-numpy.array([1,2,3,4,5,6,7,8,7,6], dtype=numpy.double));
    svm.set_epsilon(epsilon)
    svm.parallel.set_num_threads(num_threads)
    svm.train()
    
    kernel.init(feats_train, feats_test)
    out = svm.apply().get_labels()
    return out,kernel


end
if __FILE__ == $0
	puts 'SVMLight'
    classifier_svmlight_linear_term_modular(*parameter_list[0])

end
