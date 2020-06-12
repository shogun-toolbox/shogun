#!/usr/bin/env python
from tools.load import LoadMatrix
import shogun as sg
lm=LoadMatrix()

traindna = lm.load_dna('../data/fm_train_dna.dat')
testdna = lm.load_dna('../data/fm_test_dna.dat')


parameter_list = [[traindna,testdna,3,0,False],[traindna,testdna,3,0,False]]

def distance_canberraword (fm_train_dna=traindna,fm_test_dna=testdna,order=3,gap=0,reverse=False):
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

	distance = sg.create_distance("CanberraWordDistance")
	distance.init(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()
	return distance,dm_train,dm_test

if __name__=='__main__':
	print('CanberraWordDistance')
	distance_canberraword(*parameter_list[0])
