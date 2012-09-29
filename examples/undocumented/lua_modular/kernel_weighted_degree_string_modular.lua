require 'modshogun'
require 'load'

traindat = load_dna('../data/fm_train_dna.dat')
testdat = load_dna('../data/fm_test_dna.dat')

parameter_list = {{traindat,testdat,3},{traindat,testdat,20}}

function kernel_weighted_degree_string_modular (fm_train_dna,fm_test_dna,degree)

	--feats_train=modshogun.StringCharFeatures(fm_train_dna, modshogun.DNA)
	--feats_test=modshogun.StringCharFeatures(fm_test_dna, modshogun.DNA)
	--
	--kernel=modshogun.WeightedDegreeStringKernel(feats_train, feats_train, degree)
--
	--weights = {}
	--for i = degree, 1, -1 do
		--table.insert(weights, 2*i/((degree+1)*degree))
	--end
	--kernel:set_wd_weights(weights)
--
	--km_train=kernel:get_kernel_matrix()
	--kernel:init(feats_train, feats_test)
	--km_test=kernel:get_kernel_matrix()
--
	--return km_train, km_test, kernel
end

if debug.getinfo(3) == nill then
	print 'WeightedDegreeString'
	kernel_weighted_degree_string_modular(unpack(parameter_list[1]))
end
