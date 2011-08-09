require 'narray'
require 'modshogun'
require 'load'
require 'pp'

traindat = LoadMatrix.load_dna('../data/fm_train_dna.dat')
testdat = LoadMatrix.load_dna('../data/fm_test_dna.dat')
parameter_list = [[traindat,testdat,3,0,false ],[traindat,testdat,4,0,false]]

def kernel_comm_ulong_string_modular (fm_train_dna=traindat,fm_test_dna=testdat, order=3, gap=0, reverse = False)
	
	charfeat=Modshogun::StringCharFeatures.new(Modshogun::DNA)
	charfeat.set_features(fm_train_dna)
	feats_train=Modshogun::StringUlongFeatures.new(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)
	preproc=Modshogun::SortUlongString.new()
	preproc.init(feats_train)
	feats_train.add_preprocessor(preproc)
	feats_train.apply_preprocessor()


	charfeat=Modshogun::StringCharFeatures.new(Modshogun::DNA)
	charfeat.set_features(fm_test_dna)
	feats_test=Modshogun::StringUlongFeatures.new(charfeat.get_alphabet())
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)
	feats_test.add_preprocessor(preproc)
	feats_test.apply_preprocessor()

	use_sign=false

	kernel=Modshogun::CommUlongStringKernel.new(feats_train, feats_train, use_sign)

	km_train=kernel.get_kernel_matrix()
	kernel.init(feats_train, feats_test)
	km_test=kernel.get_kernel_matrix()
	pp " km_train", km_train
	pp " km_test", km_test
	return km_train,km_test,kernel
end

if __FILE__ == $0
	puts 'CommUlongString'
	kernel_comm_ulong_string_modular(*parameter_list[0])
end
