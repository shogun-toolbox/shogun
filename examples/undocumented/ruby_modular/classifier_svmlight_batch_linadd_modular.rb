# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'

train_dna=LoadMatrix.load_dna('../data/fm_train_dna.dat')
test_dna=LoadMatrix.load_dna('../data/fm_test_dna.dat')
label=LoadMatrix.load_labels('../data/label_train_dna.dat')

parameter_list=[[train_dna, test_dna, label, 20, 0.9, 1e-3, 1],
		[train_dna, test_dna, label, 20, 2.3, 1e-5, 4]]

def classifier_svmlight_batch_linadd_modular(fm_train_dna, fm_test_dna,
		label_train_dna, degree, C, epsilon, num_threads):

	try:
	except ImportError:
		print 'No support for SVMLight available.'
		return

	feats_train=StringCharFeatures(DNA)
	#feats_train.io.set_loglevel(MSG_DEBUG)
	feats_train.set_features(fm_train_dna)
	feats_test=StringCharFeatures(DNA)
	feats_test.set_features(fm_test_dna)
	degree=20

	kernel=WeightedDegreeStringKernel(feats_train, feats_train, degree)

	labels=Labels(label_train_dna)

	svm=SVMLight(C, kernel, labels)
	svm.set_epsilon(epsilon)
	svm.parallel.set_num_threads(num_threads)
	svm.train()

	kernel.init(feats_train, feats_test)

	#print 'SVMLight Objective: %f num_sv: %d' % \
	#	(svm.get_objective(), svm.get_num_support_vectors())
	svm.set_batch_computation_enabled(False)
	svm.set_linadd_enabled(False)
	svm.apply().get_labels()

	svm.set_batch_computation_enabled(True)
	labels = svm.apply().get_labels()
	return labels, svm



end
if __FILE__ == $0
	print 'SVMlight batch'
	classifier_svmlight_batch_linadd_modular(*parameter_list[0])

end
