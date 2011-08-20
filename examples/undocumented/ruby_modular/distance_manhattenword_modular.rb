# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'

traindna = LoadMatrix.load_dna('../data/fm_train_dna.dat')
testdna = LoadMatrix.load_dna('../data/fm_test_dna.dat')
testdat = LoadMatrix.load_numbers('../data/fm_test_real.dat')

parameter_list = [[traindna,testdna,testdat,3,0,False],[traindna,testdna,testdat,4,0,False]]

# *** def distance_manhattenword_modular(fm_train_dna=traindna ,fm_test_dna=testdna,fm_test_real=testdat,order=3,gap=0,reverse=False)
def distance_manhattenword_modular(fm_train_dna=traindna ,fm_test_dna=testdna,fm_test_real=testdat,order=3,gap=0,reverse=Modshogun::False.new
def distance_manhattenword_modular(fm_train_dna=traindna ,fm_test_dna=testdna,fm_test_real=testdat,order=3,gap=0,reverse.set_features)


# *** 	charfeat=StringCharFeatures(DNA)
	charfeat=Modshogun::StringCharFeatures.new
	charfeat.set_features(DNA)
	charfeat.set_features(fm_train_dna)
# *** 	feats_train=StringWordFeatures(charfeat.get_alphabet())
	feats_train=Modshogun::StringWordFeatures.new
	feats_train.set_features(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)
# *** 	preproc=SortWordString()
	preproc=Modshogun::SortWordString.new
	preproc.set_features()
	preproc.init(feats_train)
	feats_train.add_preprocessor(preproc)
	feats_train.apply_preprocessor()

# *** 	charfeat=StringCharFeatures(DNA)
	charfeat=Modshogun::StringCharFeatures.new
	charfeat.set_features(DNA)
	charfeat.set_features(fm_test_dna)
# *** 	feats_test=StringWordFeatures(charfeat.get_alphabet())
	feats_test=Modshogun::StringWordFeatures.new
	feats_test.set_features(charfeat.get_alphabet())
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)
	feats_test.add_preprocessor(preproc)
	feats_test.apply_preprocessor()

# *** 	distance=ManhattanWordDistance(feats_train, feats_train)
	distance=Modshogun::ManhattanWordDistance.new
	distance.set_features(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()
	return dm_train,dm_test


end
if __FILE__ == $0
	puts 'ManhattanWordDistance'
	distance_manhattenword_modular(*parameter_list[0])

end
