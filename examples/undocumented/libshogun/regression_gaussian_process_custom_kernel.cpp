/*
* Copyright (c) The Shogun Machine Learning Toolbox
* Written (w) 2014 pl8787
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The views and conclusions contained in the software and documentation are those
* of the authors and should not be interpreted as representing official policies,
* either expressed or implied, of the Shogun Development Team.
*/

#include <shogun/lib/config.h>

// Eigen3 is required for working with this example
#ifdef HAVE_EIGEN3

#include <shogun/base/init.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/IndexFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/mathematics/Math.h>
#include <shogun/machine/gp/ExactInferenceMethod.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/evaluation/MeanSquaredError.h>

using namespace shogun;

// Create some easy regression data: 1d noisy sine wave
void generate_data(SGMatrix<float64_t> &X, SGVector<float64_t> &Y, index_t n)
{
	float64_t x_range = 6;

	X = SGMatrix<float64_t>(1, n);
	Y = SGVector<float64_t>(n);

	for (index_t  i=0; i<n; ++i)
	{
		X[i] = CMath::random(0.0, x_range);
		Y[i] = CMath::sin(X[i]);
	}
}

// Do the GP regression with custom kernel
void gp_regression_custom_kernel()
{
	// Sample count
	index_t n = 200;
	// Generate dataset
	SGMatrix<float64_t> X;
	SGVector<float64_t> Y;
	generate_data(X, Y, n);
	SGVector<index_t> feat_idx_train_v(n);
	SGVector<index_t> feat_idx_test_v(n);
	feat_idx_train_v.range_fill();
	feat_idx_test_v.range_fill();
	CIndexFeatures* feat_idx_train = new CIndexFeatures(feat_idx_train_v);
	CIndexFeatures* feat_idx_test = new CIndexFeatures(feat_idx_test_v);
	SG_REF(feat_idx_train);
	SG_REF(feat_idx_test);

	// Convert training and testing data into shogun representation
	CDenseFeatures<float64_t>* feat = new CDenseFeatures<float64_t>(X);
	CRegressionLabels* lab_train = new CRegressionLabels(Y);
	CRegressionLabels* lab_test = new CRegressionLabels(Y);
	SG_REF(feat);
	SG_REF(lab_train);
	SG_REF(lab_test);

	// GP parameters
	float64_t sigma2 = 1;
	float64_t kernel_log_sigma = 1;
	float64_t kernel_log_scale = 1;

	// Allocate our mean function
	CZeroMean* mean = new CZeroMean();
	SG_REF(mean);

	// Allocate our likelihood function
	CGaussianLikelihood* lik = new CGaussianLikelihood();
	SG_REF(lik);
	lik->set_sigma(CMath::sqrt(sigma2));

	// Allocate our Kernel
	float64_t kernel_sigma = 2*CMath::exp(2*kernel_log_sigma);
	CGaussianKernel* gaussian_kernel = new CGaussianKernel(10, kernel_sigma);
	SG_REF(gaussian_kernel);
	gaussian_kernel->init(feat, feat);

	CCustomKernel* kernel = new CCustomKernel(gaussian_kernel);
	SG_REF(kernel);

	// Calculate mean squared error of train and test
	CMeanSquaredError* eval = new CMeanSquaredError();
	SG_REF(eval);

	// Allocate our inference method
	CExactInferenceMethod* inf = new CExactInferenceMethod(kernel,
						  feat_idx_train, mean, lab_train, lik);
	SG_REF(inf);
	// Parameter of kernel scale
	inf->set_scale(CMath::exp(kernel_log_scale));

	// Finally use these to allocate the Gaussian Process Object
	CGaussianProcessRegression* gpr = new CGaussianProcessRegression(inf);
	SG_REF(gpr);

	SGVector<index_t> train_idx_v(180);
	SGVector<index_t> test_idx_v(20);

	train_idx_v.range_fill(0);
	test_idx_v.range_fill(180);

	train_idx_v.display_vector("Train Index");
	test_idx_v.display_vector("Test Index");

	feat_idx_train->add_subset(train_idx_v);
	lab_train->add_subset(train_idx_v);
	feat_idx_test->add_subset(test_idx_v);
	lab_test->add_subset(test_idx_v);

	// perform inference on train
	CRegressionLabels* predictions_train=gpr->apply_regression(feat_idx_train);
	SG_REF(predictions_train);
	float64_t error_train = eval->evaluate(predictions_train, lab_train);

	// perform inference on test
	CRegressionLabels* predictions_test=gpr->apply_regression(feat_idx_test);
	SG_REF(predictions_test);
	float64_t error_test = eval->evaluate(predictions_test, lab_test);

	feat_idx_train->remove_all_subsets();
	feat_idx_test->remove_all_subsets();
	lab_train->remove_all_subsets();
	lab_test->remove_all_subsets();

	SG_SPRINT("Train Error:%f, Test Error:%f\n", error_train, error_test);

	SG_UNREF(predictions_train);
	SG_UNREF(predictions_test);

	SG_UNREF(mean);
	SG_UNREF(lik);
	SG_UNREF(gaussian_kernel);
	SG_UNREF(kernel);
	SG_UNREF(eval);
	SG_UNREF(feat);
	SG_UNREF(lab_train);
	SG_UNREF(lab_test);
	SG_UNREF(feat_idx_train);
	SG_UNREF(feat_idx_test);
	SG_UNREF(inf);
	SG_UNREF(gpr);
}

// Main of this GP with simple custom kernel
int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	gp_regression_custom_kernel();

	exit_shogun();
	return 0;
}

#endif /* HAVE_EIGEN3 */
