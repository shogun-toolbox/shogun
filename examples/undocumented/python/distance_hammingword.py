#!/usr/bin/env python
from tools.load import LoadMatrix
lm=LoadMatrix()

traindna = lm.load_dna('../data/fm_train_dna.dat')
testdna = lm.load_dna('../data/fm_test_dna.dat')
testdat = lm.load_labels('../data/fm_test_real.dat')

parameter_list = [[traindna,testdna,testdat,4,0,False,False],
		[traindna,testdna,testdat,3,0,False,False]]

def distance_hammingword (fm_train_dna=traindna,fm_test_dna=testdna,
		fm_test_real=testdat,order=3,gap=0,reverse=False,use_sign=False):

	from shogun import StringCharFeatures, StringWordFeatures, DNA
	from shogun import SortWordString
	from shogun import HammingWordDistance

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_train_dna)
	feats_train=StringWordFeatures(charfeat.get_alphabet())
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse)
	preproc=SortWordString()
	preproc.fit(feats_train)
	feats_train = preproc.apply(feats_train)

	charfeat=StringCharFeatures(DNA)
	charfeat.set_features(fm_test_dna)
	feats_test=StringWordFeatures(charfeat.get_alphabet())
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse)
	feats_test = preproc.apply(feats_test)

	distance=HammingWordDistance(feats_train, feats_train, use_sign)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()
	return distance,dm_train,dm_test

if __name__=='__main__':
	print('HammingWordDistance')
	distance_hammingword(*parameter_list[0])
