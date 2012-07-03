###########################################################################
# Mean prediction from Gaussian Processes based on classifier_libsvm_minimal_modular.py
###########################################################################
from numpy import *
from numpy.random import randn
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *
from shogun.Regression import *

num=100
dist=1
width=2.1
C=1

traindata_real=concatenate((randn(2,num)-dist, randn(2,num)+dist), axis=1);
testdata_real=concatenate((randn(2,num)-dist, randn(2,num)+dist), axis=1);

trainlab = concatenate((-ones(num), ones(num)));
labels=RegressionLabels(trainlab);

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);
kernel=GaussianKernel(feats_train, feats_train, width);
zmean = ZeroMean();
lik = GaussianLikelihood();
inf = ExactInferenceMethod(kernel, feats_train, zmean, labels, lik);
gp = GaussianProcessRegression(inf, feats_train, labels);

alpha = inf.get_alpha();
diagonal = inf.get_diagonal_vector();
cholesky = inf.get_cholesky();
gp.set_return_type(GaussianProcessRegression.GP_RETURN_COV);

covariance = gp.apply_regression(feats_test);

gp.set_return_type(GaussianProcessRegression.GP_RETURN_MEANS);

predictions = gp.apply_regression();

testerr=mean(sign(out)!=testlab)

print("Alpha Vector\n");
alpha

print("Labels\n");
labels.get_labels()

print("sW Matrix");
diagonal

print("Covariances");
covariance.get_labels()

print("Mean Predictions");
predictions.get_labels()

print("Cholesky Matrix L");
cholesky



