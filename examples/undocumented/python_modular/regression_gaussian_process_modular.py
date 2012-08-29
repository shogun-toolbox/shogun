from numpy import concatenate, ones
from numpy.random import randn
num=100
dist=1

traindat=concatenate((randn(2,num)-dist, randn(2,num)+dist), axis=1)
testdat=concatenate((randn(2,num)-dist, randn(2,num)+dist), axis=1)
label_traindat = concatenate((-ones(num), ones(num)))

parameter_list=[[traindat, testdat, label_traindat, 2.1]]

def regression_gaussian_process_modular (traindata_real=traindat, \
		testdata_real=testdat, \
		trainlab=label_traindat, width=2.1):
	from numpy.random import randn
	from shogun.Features import RealFeatures, RegressionLabels
	from shogun.Kernel import GaussianKernel
	try:
		from shogun.Regression import GaussianLikelihood, ZeroMean, \
				ExactInferenceMethod, GaussianProcessRegression
	except ImportError:
		print "Eigen3 needed for Gaussian Processes"
		return

	labels=RegressionLabels(trainlab)

	feats_train=RealFeatures(traindata_real)
	feats_test=RealFeatures(testdata_real)
	kernel=GaussianKernel(feats_train, feats_train, width)
	zmean = ZeroMean()
	lik = GaussianLikelihood()
	inf = ExactInferenceMethod(kernel, feats_train, zmean, labels, lik)
	gp = GaussianProcessRegression(inf, feats_train, labels)

	alpha = inf.get_alpha()
	diagonal = inf.get_diagonal_vector()
	cholesky = inf.get_cholesky()
	gp.set_return_type(GaussianProcessRegression.GP_RETURN_COV)

	covariance = gp.apply_regression(feats_test)

	gp.set_return_type(GaussianProcessRegression.GP_RETURN_MEANS)

	predictions = gp.apply_regression()

	print("Alpha Vector")
	print(alpha)

	print("Labels")
	print(labels.get_labels())

	print("sW Matrix")
	print(diagonal)

	print("Covariances")
	print(covariance.get_labels())

	print("Mean Predictions")
	print(predictions.get_labels())

	print("Cholesky Matrix L")
	print(cholesky)
	return gp, alpha, labels, diagonal, covariance, predictions, cholesky

if __name__=='__main__':
	print('Gaussian Process Regression')
	regression_gaussian_process_modular(*parameter_list[0])
