require 'nmatrix'
require 'modshogun'
require 'pp'

require_relative 'load'

###########################################################################
# anova kernel
###########################################################################

traindat = LoadMatrix.load_numbers('../data/fm_train_real.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')
parameter_list = [[traindat,testdat,2,10], [traindat,testdat,5,10]]

def kernel_anova_modular(fm_train_real=traindat,fm_test_real=testdat,cardinality=2, size_cache=10)
	
	feats_train=Modshogun::RealFeatures.new
	feats_train.set_feature_matrix(fm_train_real)
	feats_test=Modshogun::RealFeatures.new
	feats_test.set_feature_matrix(fm_test_real)
	
	kernel=Modshogun::ANOVAKernel.new(feats_train, feats_train, cardinality, size_cache)
        
	for i in 0..feats_train.get_num_vectors
		for j in 0..feats_train.get_num_vectors
			k1 = kernel.compute_rec1(i,j)
			k2 = kernel.compute_rec2(i,j)
			if (k1-k2).abs > 1e-10
				puts "|#{k1}|#{k2}|"
			end
		end
	end

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	return km_train, km_test, kernel
end

if __FILE__ == $0
	puts 'ANOVA'
	pp kernel_anova_modular(*parameter_list[0])
end
