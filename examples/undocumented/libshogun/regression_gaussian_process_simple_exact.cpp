/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Roman Votyakov, Soeren Sonnenburg, Pan Deng
 */

#ifdef USE_GPL_SHOGUN

#include <shogun/lib/config.h>

#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/machine/gp/ExactInferenceMethod.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/base/init.h>

using namespace shogun;

void test()
{
	/* create some easy regression data: 1d noisy sine wave */
	index_t n=100;
	float64_t x_range=6;

	SGMatrix<float64_t> X(1, n);
	SGMatrix<float64_t> X_test(1, n);
	SGVector<float64_t> Y(n);

	for (index_t  i=0; i<n; ++i)
	{
		X[i]=Math::random(0.0, x_range);
		X_test[i]=(float64_t)i / n*x_range;
		Y[i] = std::sin(X[i]);
	}

	/* shogun representation */
	DenseFeatures<float64_t>* feat_train=new DenseFeatures<float64_t>(X);
	DenseFeatures<float64_t>* feat_test=new DenseFeatures<float64_t>(X_test);
	RegressionLabels* label_train=new RegressionLabels(Y);

	/* specity GPR with exact inference */
	float64_t sigma=1;
	float64_t shogun_sigma=sigma*sigma*2;
	GaussianKernel* kernel=new GaussianKernel(10, shogun_sigma);
	ZeroMean* mean=new ZeroMean();
	GaussianLikelihood* lik=new GaussianLikelihood();
	lik->set_sigma(1);
	ExactInferenceMethod* inf=new ExactInferenceMethod(kernel, feat_train,
			mean, label_train, lik);
	GaussianProcessRegression* gpr=new GaussianProcessRegression(inf);

	/* perform inference */
	RegressionLabels* predictions=gpr->apply_regression(feat_test);
	predictions->get_labels().display_vector("predictions");

}

int main(int argc, char** argv)
{
	init_shogun_with_defaults();

	test();

	exit_shogun();
	return 0;
}

#else //USE_GPL_SHOGUN
int main(int argc, char** argv)
{
	return 0;
}
#endif //USE_GPL_SHOGUN
