#!/usr/bin/env python
traindna = '../data/fm_train_dna.dat'
testdna = '../data/fm_test_dna.dat'

parameter_list = [[traindna,testdna,testdat,3,0,False],[traindna,testdna,testdat,4,0,False]]

def distance_manhattenword_modular (train_fname=traindna ,test_fname=testdna,order=3,gap=0,reverse=False):

	from modshogun import StringCharFeatures, StringWordFeatures, DNA
	from modshogun import SortWordString, ManhattanWordDistance, CSVFile

	charfeat=StringCharFeatures(CSVFile(train_fname), DNA)
	feats_train=StringWordFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)
	preproc=SortWordString()
	preproc.init(feats_train)
	feats_train.add_preprocessor(preproc)
	feats_train.apply_preprocessor()

	charfeat=StringCharFeatures(CSVFile(test_fname), DNA)
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
