#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()

traindna = lm.load_dna('../data/fm_train_dna.dat')
testdna = lm.load_dna('../data/fm_test_dna.dat')
testdat = lm.load_labels('../data/fm_test_real.dat')

parameter_list = [[traindna,testdna,testdat,4,0,False,False],
		[traindna,testdna,testdat,3,0,False,False]]

def distance_hammingword_modular (fm_train_dna=traindna,fm_test_dna=testdna,
		fm_test_real=testdat,order=3,gap=0,reverse=False,use_sign=False):

	from modshogun import StringCharFeatures, StringWordFeatures, DNA
	from modshogun import SortWordString
	from modshogun import HammingWordDistance

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_train_dna)
	feats_train=StringWordFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)
	preproc=SortWordString()
	preproc.init(feats_train)
	feats_train.add_preprocessor(preproc)
	feats_train.apply_preprocessor()

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_test_dna)
	feats_test=StringWordFeatures(charfeat.get_alphabet())
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)
	feats_test.add_preprocessor(preproc)
	feats_test.apply_preprocessor()

	distance=HammingWordDistance(feats_train, feats_train, use_sign)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()
	return distance,dm_train,dm_test

if __name__=='__main__':
	print('HammingWordDistance')
	distance_hammingword_modular(*parameter_list[0])
