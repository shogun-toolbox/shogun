'''
This example shows how to use the class MahalanobisDistance to compute the 
distance from a feature vector to another vector of the same distributiona or the 
distance from a vector to a distribution of feature vectors.

The program first loads toy data and creates objects of the class RealFeatures
to create an instance of the class MahalanobisDistance. Later it computes and 
prints on the screen the Mahalanobis distance from each of the feature vectors 
in feats_test to the set of features feats_train.

Note that if both distributions are equal thenonly the second argument is relevant 
in the method distance of the class MahalanobisDistance.

If you compare the results given by this program with other software, please 
keep in mind that the covariance matrix of a group of features in shogun is
computed using the number of features (N) as normalizer while other software
(e.g. octave) may use (N-1) as normalizer instead.
'''
  
from tools.load import LoadMatrix
lm = LoadMatrix()

traindat = lm.load_numbers('../data/fm_train_real.dat')
testdat  = lm.load_numbers('../data/fm_test_real.dat')

parameter_list = [[traindat, testdat]]

def distance_mahalanobis_modular (fm_train_real = traindat, fm_test_real = testdat):

	from shogun.Features import RealFeatures
	from shogun.Distance import MahalanobisDistance

	feats_train = RealFeatures(fm_train_real)
	feats_test  = RealFeatures(fm_test_real)

	distance = MahalanobisDistance(feats_test, feats_train)
	for i in range(feats_test.get_num_vectors()):
		for j in range(feats_train.get_num_vectors()):
			dm = distance.distance(i, j)
			print dm

if __name__=='__main__':
	print 'MahalanobisDistance'
	distance_mahalanobis_modular(*parameter_list[0])
