from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()

traindna=lm.load_dna('../data/fm_train_dna.dat')
testdna=lm.load_dna('../data/fm_test_dna.dat')
parameter_list=[[traindna,testdna,3,0,'n'],[traindna,testdna,4,0,'n']]

def distance_canberraword (fm_train_dna=traindna,fm_test_dna=testdna,order=3,
			    gap=0,reverse='n'):

	sg('set_distance', 'CANBERRA', 'WORD')
	sg('add_preproc', 'SORTWORDSTRING')
	sg('set_features', 'TRAIN', fm_train_dna, 'DNA')
	sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
	sg('attach_preproc', 'TRAIN')
	dm=sg('get_distance_matrix', 'TRAIN')
	sg('set_features', 'TEST', fm_test_dna, 'DNA')
	sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse)
	sg('attach_preproc', 'TEST')
	dm=sg('get_distance_matrix', 'TEST')
	return dm

if __name__=='__main__':
	print('CanberraWordDistance')
	distance_canberraword(*parameter_list[0])
