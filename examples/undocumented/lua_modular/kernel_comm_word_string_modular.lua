require 'shogun'
require 'load'

traindat = load_dna('../data/fm_train_dna.dat')
testdat = load_dna('../data/fm_test_dna.dat')
parameter_list = {{traindat,testdat,4,0,false, false},{traindat,testdat,4,0,False,False}}

function kernel_comm_word_string_modular (fm_train_dna,fm_test_dna, order, gap, reverse, use_sign)
	charfeat=StringCharFeatures(DNA)
	charfeat:set_features(fm_train_dna)
	feats_train=StringWordFeatures(charfeat:get_alphabet())
	feats_train:obtain_from_char(charfeat, order-1, order, gap, reverse)
	
	preproc=SortWordString()
	preproc:init(feats_train)
	feats_train:add_preprocessor(preproc)
	feats_train:apply_preprocessor()

	charfeat=StringCharFeatures(DNA)
	charfeat:set_features(fm_test_dna)
	feats_test=StringWordFeatures(charfeat:get_alphabet())
	feats_test:obtain_from_char(charfeat, order-1, order, gap, reverse)
	feats_test:add_preprocessor(preproc)
	feats_test:apply_preprocessor()

	kernel=CommWordStringKernel(feats_train, feats_train, use_sign)

	km_train=kernel:get_kernel_matrix()
	kernel:init(feats_train, feats_test)
	km_test=kernel:get_kernel_matrix()
	return km_train,km_test,kernel
end

if debug.getinfo(3) == nill then
	print 'CommWordString'
	kernel_comm_word_string_modular(unpack(parameter_list[1]))
end
