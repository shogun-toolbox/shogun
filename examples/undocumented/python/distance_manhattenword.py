#!/usr/bin/env python
traindna = '../data/fm_train_dna.dat'
testdna = '../data/fm_test_dna.dat'

parameter_list = [[traindna,testdna,3,0,False],[traindna,testdna,4,0,False]]

def distance_manhattenword (train_fname=traindna,test_fname=testdna,order=3,gap=0,reverse=False):
	from shogun import StringCharFeatures, StringWordFeatures, DNA
	from shogun import SortWordString, ManhattanWordDistance, CSVFile

	charfeat=StringCharFeatures(CSVFile(train_fname), DNA)
	feats_train=StringWordFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)
	preproc = SortWordString()
	preproc.fit(feats_train)
	feats_train = preproc.transform(feats_train)

	charfeat=StringCharFeatures(CSVFile(test_fname), DNA)
	feats_test=StringWordFeatures(charfeat.get_alphabet())
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)
	feats_test = preproc.transform(feats_test)

	distance=ManhattanWordDistance(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()
	return dm_train,dm_test

if __name__=='__main__':
	print('ManhattanWordDistance')
	distance_manhattenword(*parameter_list[0])
