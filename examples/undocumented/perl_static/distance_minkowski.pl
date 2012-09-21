from tools.load import LoadMatrix
from sg import sg
lm=LoadMatrix()


traindat=lm.load_numbers('../data/fm_train_real.dat')
testdat=lm.load_numbers('../data/fm_test_real.dat')
parameter_list=[[traindat,testdat,3.],[traindat,testdat,4.]]
def distance_minkowski (fm_train_real=traindat,fm_test_real=testdat,k=3.):
	sg('set_distance', 'MINKOWSKI', 'REAL', k)

	sg('set_features', 'TRAIN', fm_train_real)
	dm=sg('get_distance_matrix', 'TRAIN')

	sg('set_features', 'TEST', fm_test_real)
	dm=sg('get_distance_matrix', 'TEST')
	return dm

if __name__=='__main__':
	print('MinkowskiMetric')
	distance_minkowski(*parameter_list[0])
