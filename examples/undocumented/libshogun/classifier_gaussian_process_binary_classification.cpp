/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Roman Votyakov, Heiko Strathmann, Wu Lin, Pan Deng, Bjoern Esser
 */

#ifdef USE_GPL_SHOGUN

#include <shogun/lib/config.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/machine/gp/SingleLaplacianInferenceMethod.h>
#include <shogun/machine/gp/EPInferenceMethod.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/LogitLikelihood.h>
#include <shogun/machine/gp/ProbitLikelihood.h>
#include <shogun/classifier/GaussianProcessClassification.h>
#include <shogun/io/CSVFile.h>

using namespace shogun;

// files with training data
const char* fname_feat_train="../data/fm_train_real.dat";
const char* fname_label_train="../data/label_train_twoclass.dat";

// file with testing data
const char* fname_feat_test="../data/fm_test_real.dat";

int main(int argc, char** argv)
{
	// trainig data
	SGMatrix<float64_t> X_train;
	SGVector<float64_t> y_train;

	// load training features from file
	CSVFile* file_feat_train=new CSVFile(fname_feat_train);
	X_train.load(file_feat_train);

	// load training labels from file
	CSVFile* file_label_train=new CSVFile(fname_label_train);
	y_train.load(file_label_train);

	// testing features
	SGMatrix<float64_t> X_test;

	// load testing features from file
	CSVFile* file_feat_test=new CSVFile(fname_feat_test);
	X_test.load(file_feat_test);

	// convert training and testing data into shogun representation
	DenseFeatures<float64_t>* feat_train=new DenseFeatures<float64_t>(X_train);
	BinaryLabels* lab_train=new BinaryLabels(y_train);
	DenseFeatures<float64_t>* feat_test=new DenseFeatures<float64_t>(X_test);

	// create Gaussian kernel with width = 2.0
	GaussianKernel* kernel=new GaussianKernel(10, 2.0);

	// create zero mean function
	ZeroMean* mean=new ZeroMean();

	// you can easily switch between probit and logit likelihood models
    // by uncommenting/commenting the following lines:

	// create probit likelihood model
	// ProbitLikelihood* lik=new ProbitLikelihood();

	// create logit likelihood model
	LogitLikelihood* lik=new LogitLikelihood();

	// you can easily switch between SingleLaplace and EP approximation by
	// uncommenting/commenting the following lines:

	// specify SingleLaplace approximation inference method
	// CSingleLaplacianInferenceMethod* inf=new CSingleLaplacianInferenceMethod(kernel,
	//		feat_train, mean, lab_train, lik);

	// specify EP approximation inference method
	EPInferenceMethod* inf=new EPInferenceMethod(kernel, feat_train, mean,
			lab_train, lik);

	// create and train GP classifier, which uses SingleLaplace approximation
	GaussianProcessClassification* gpc=new GaussianProcessClassification(inf);
	gpc->train();

	// apply binary classification to the test data and get -1/+1
	// labels of the predictions
	BinaryLabels* predictions=gpc->apply_binary(feat_test);
	predictions->get_labels().display_vector("predictions");

	// get probabilities p(y*=1|x*) for each testing feature x*
	SGVector<float64_t> p_test=gpc->get_probabilities(feat_test);
	p_test.display_vector("predictive probability");

	// get predictive mean
	SGVector<float64_t> mu_test=gpc->get_mean_vector(feat_test);
	mu_test.display_vector("predictive mean");

	// get predictive variance
	SGVector<float64_t> s2_test=gpc->get_variance_vector(feat_test);
	s2_test.display_vector("predictive variance");

	// free up memory

	return 0;
}
#else //USE_GPL_SHOGUN
int main(int argc, char** argv)
{
	return 0;
}
#endif //USE_GPL_SHOGUN
