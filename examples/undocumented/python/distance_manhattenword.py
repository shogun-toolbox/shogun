#!/usr/bin/env python
import shogun as sg
traindna = '../data/fm_train_dna.dat'
testdna = '../data/fm_test_dna.dat'

parameter_list = [[traindna,testdna,3,0,False],[traindna,testdna,4,0,False]]

def distance_manhattenword (train_fname=traindna,test_fname=testdna,order=3,gap=0,reverse=False):

	charfeat=sg.create_string_features(sg.read_csv(train_fname), sg.DNA)
	feats_train=sg.create_string_features(charfeat, order-1, order, gap, reverse)
	feats_train.put("alphabet", sg.as_alphabet(charfeat.get("alphabet")))
	preproc = sg.create_transformer("SortWordString")
	preproc.fit(feats_train)
	feats_train = preproc.transform(feats_train)

	charfeat=sg.create_string_features(sg.read_csv(test_fname), sg.DNA)
	feats_test=sg.create_string_features(charfeat, order-1, order, gap, reverse)
	feats_test.put("alphabet", sg.as_alphabet(charfeat.get("alphabet")))
	feats_test = preproc.transform(feats_test)

	distance = sg.create_distance('ManhattanWordDistance')
	distance.init(feats_train, feats_train)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()
	return dm_train,dm_test

if __name__=='__main__':
	print('ManhattanWordDistance')
	distance_manhattenword(*parameter_list[0])
