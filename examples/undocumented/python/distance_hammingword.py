#!/usr/bin/env python
import shogun as sg
from tools.load import LoadMatrix
lm=LoadMatrix()

traindna = lm.load_dna('../data/fm_train_dna.dat')
testdna = lm.load_dna('../data/fm_test_dna.dat')
testdat = lm.load_labels('../data/fm_test_real.dat')

parameter_list = [[traindna,testdna,testdat,4,0,False,False],
		[traindna,testdna,testdat,3,0,False,False]]

def distance_hammingword (fm_train_dna=traindna,fm_test_dna=testdna,
		fm_test_real=testdat,order=3,gap=0,reverse=False,use_sign=False):

	charfeat=sg.create_string_features(fm_train_dna, sg.DNA)
	feats_train=sg.create_string_features(charfeat, order-1, order, gap, reverse)
	feats_train.put("alphabet", sg.as_alphabet(charfeat.get("alphabet")))
	preproc = sg.create_transformer("SortWordString")
	preproc.fit(feats_train)
	feats_train = preproc.transform(feats_train)

	charfeat=sg.create_string_features(fm_test_dna, sg.DNA)
	feats_test=sg.create_string_features(charfeat, order-1, order, gap, reverse)
	feats_test.put("alphabet", sg.as_alphabet(charfeat.get("alphabet")))
	feats_test = preproc.transform(feats_test)

	distance = sg.create_distance("HammingWordDistance", use_sign=use_sign)
	distance.init(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()
	return distance,dm_train,dm_test

if __name__=='__main__':
	print('HammingWordDistance')
	distance_hammingword(*parameter_list[0])
