# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'
traindat = LoadMatrix.load_dna('../data/fm_train_dna.dat')
testdat = LoadMatrix.load_dna('../data/fm_test_dna.dat')

parameter_list = [[traindat,testdat,5,5],[traindat,testdat,6,6]]

def kernel_distantsegments_modular(fm_train_dna=traindat,fm_test_dna=testdat,delta=5, theta=5)

	feats_train=StringCharFeatures(fm_train_dna, DNA)
	feats_test=StringCharFeatures(fm_test_dna, DNA)
	
	kernel=DistantSegmentsKernel(feats_train, feats_train, 10, delta, theta)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

	return km_train, km_test, kernel



end
if __FILE__ == $0
	print 'DistantSegments'
	kernel_distantsegments_modular(*parameter_list[0])

end
