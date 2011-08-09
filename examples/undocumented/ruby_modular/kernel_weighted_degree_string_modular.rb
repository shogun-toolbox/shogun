# this was trancekoded by the awesome trancekoder
require 'narray'
require 'modshogun'
require 'load'
require 'pp'
traindat = LoadMatrix.load_dna('../data/fm_train_dna.dat')
testdat = LoadMatrix.load_dna('../data/fm_test_dna.dat')

parameter_list = [[traindat,testdat,3],[traindat,testdat,20]]

def kernel_weighted_degree_string_modular(fm_train_dna=traindat,fm_test_dna=testdat,degree=20)

	feats_train=StringCharFeatures(fm_train_dna, DNA)
	#feats_train.io.set_loglevel(MSG_DEBUG)
	feats_test=StringCharFeatures(fm_test_dna, DNA)
	
	kernel=WeightedDegreeStringKernel(feats_train, feats_train, degree)

	weights=arange(1,degree+1,dtype=double)[::-1]/ \
		sum(arange(1,degree+1,dtype=double))
	kernel.set_wd_weights(weights)
	#kernel.set_position_weights(ones(len(fm_train_dna[0]), dtype=float64))

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()

    #this is how to serializate the kernel
	#import pickle
	#pickle.dump(kernel, file('kernel_obj.dump','w'), protocol=2)
	#k=pickle.load(file('kernel_obj.dump','r'))


	return km_train, km_test, kernel



end
if __FILE__ == $0
	print 'WeightedDegreeString'
	kernel_weighted_degree_string_modular(*parameter_list[0])

end
