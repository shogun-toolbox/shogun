#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()

traindna = lm.load_dna('../data/fm_train_dna.dat')
testdna = lm.load_dna('../data/fm_test_dna.dat')
testdat = lm.load_numbers('../data/fm_test_real.dat')

parameter_list = [[traindna,testdna,testdat,3,0,False],[traindna,testdna,testdat,4,0,False]]

def distance_manhattenword_modular (fm_train_dna=traindna ,fm_test_dna=testdna,fm_test_real=testdat,order=3,gap=0,reverse=False):

	from modshogun import StringCharFeatures, StringWordFeatures, DNA
	from modshogun import SortWordString
	from modshogun import ManhattanWordDistance

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

	distance=ManhattanWordDistance(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()
	return dm_train,dm_test

if __name__=='__main__':
	print('ManhattanWordDistance')
	distance_manhattenword_modular(*parameter_list[0])
