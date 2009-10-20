def norm_squared_distance ():
	from shogun.Features import RealFeatures
	from shogun.Distance import EuclidianDistance
	print 'EuclidianDistance - NormSquared'

	feats_train=RealFeatures(fm_train_real)
	feats_test=RealFeatures(fm_test_real)

	distance=EuclidianDistance(feats_train, feats_train)
	distance.set_disable_sqrt(True)

	dm_train=distance.get_distance_matrix()
	distance.init(feats_train, feats_test)
	dm_test=distance.get_distance_matrix()

if __name__=='__main__':
	from tools.load import LoadMatrix
	lm=LoadMatrix()
	fm_train_real=lm.load_numbers('../data/fm_train_real.dat')
	fm_test_real=lm.load_numbers('../data/fm_test_real.dat')
	norm_squared_distance()
