require 'modshogun'
require 'load'

traindat = load_dna('../data/fm_train_dna.dat')
testdat =  load_dna('../data/fm_test_dna.dat')
parameter_list = {{traindat,testdat,3,0,false},{traindat,testdat,4,0,false}}

function kernel_comm_ulong_string_modular (fm_train_dna,fm_test_dna, order, gap, reverse)
	--charfeat=modshogun.StringCharFeatures(modshogun.DNA)
	--charfeat:set_features(fm_train_dna)
	--feats_train=modshogun.StringUlongFeatures(charfeat:get_alphabet())
	--feats_train:obtain_from_char(charfeat, order-1, order, gap, reverse)
	--preproc=modshogun.SortUlongString()
	--preproc:init(feats_train)
	--feats_train:add_preprocessor(preproc)
	--feats_train:apply_preprocessor()


	--charfeat=modshogun.StringCharFeatures(modshogun.DNA)
	--charfeat:set_features(fm_test_dna)
	--feats_test=modshogun.StringUlongFeatures(charfeat:get_alphabet())
	--feats_test:obtain_from_char(charfeat, order-1, order, gap, reverse)
	--feats_test:add_preprocessor(preproc)
	--feats_test:apply_preprocessor()
	--use_sign=false
	--kernel=modshogun.CommUlongStringKernel(feats_train, feats_train, use_sign)
	--km_train=kernel:get_kernel_matrix()
	--kernel:init(feats_train, feats_test)
	--km_test=kernel:get_kernel_matrix()
	--return km_train,km_test,kernel
end

if debug.getinfo(3) == nill then
	print 'CommUlongString'
	kernel_comm_ulong_string_modular(unpack(parameter_list[1]))
end
