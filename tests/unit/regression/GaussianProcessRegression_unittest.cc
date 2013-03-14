#include <shogun/lib/config.h>
#ifdef HAVE_EIGEN3

#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/regression/gp/ExactInferenceMethod.h>
#include <shogun/regression/gp/ZeroMean.h>
#include <shogun/regression/gp/GaussianLikelihood.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(GaussianProcessRegression,apply_regression)
{
	/* create some easy regression data: 1d noisy sine wave */
	index_t n=3;

	SGMatrix<float64_t> X(1, n);
	SGMatrix<float64_t> X_test(1, n);
	SGVector<float64_t> Y(n);

	X[0]=0;
	X[1]=1.1;
	X[2]=2.2;

	X_test[0]=0.3;
	X_test[1]=1.3;
	X_test[2]=2.5;

	for (index_t i=0; i<n; ++i)
	{
		Y[i]=CMath::sin(X(0, i));
	}

	/* shogun representation */
	CDenseFeatures<float64_t>* feat_train=new CDenseFeatures<float64_t>(X);
	CDenseFeatures<float64_t>* feat_test=new CDenseFeatures<float64_t>(X_test);
	CRegressionLabels* label_train=new CRegressionLabels(Y);

	/* specity GPR with exact inference */
	float64_t sigma=1;
	float64_t shogun_sigma=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, shogun_sigma);
	CZeroMean* mean=new CZeroMean();
	CGaussianLikelihood* lik=new CGaussianLikelihood();
	lik->set_sigma(1);
	CExactInferenceMethod* inf=new CExactInferenceMethod(kernel, feat_train,
			mean, label_train, lik);
	CGaussianProcessRegression* gpr=new CGaussianProcessRegression(inf,
			feat_train, label_train);

	/* perform inference */
	gpr->set_return_type(CGaussianProcessRegression::GP_RETURN_MEANS);
	CRegressionLabels* predictions=gpr->apply_regression(feat_test);

	/* do some checks against gpml toolbox*/

	/* m =

   0.221198406887592
   0.537437461176145
   0.431605035301329 */
	SGVector<float64_t> prediction_vector=predictions->get_labels();
	EXPECT_LE(CMath::abs(prediction_vector[0]-0.221198406887592), 10E-15);
	EXPECT_LE(CMath::abs(prediction_vector[1]-0.537437461176145), 10E-15);
	EXPECT_LE(CMath::abs(prediction_vector[2]-0.431605035301329), 10E-15);

	SG_UNREF(predictions);
	SG_UNREF(gpr);
}

TEST(GaussianProcessRegression,apply_regression_larger_test)
{
	/* create some easy regression data: 1d noisy sine wave */
	index_t n=3;
	index_t n_test=5;

	SGMatrix<float64_t> X(1, n);
	SGMatrix<float64_t> X_test(1, n_test);
	SGVector<float64_t> Y(n);

	X[0]=0;
	X[1]=1.1;
	X[2]=2.2;

	X_test[0]=0.3;
	X_test[1]=1.3;
	X_test[2]=2.5;
	X_test[2]=2.7;
	X_test[2]=3.1;


	for (index_t i=0; i<n; ++i)
	{
		Y[i]=CMath::sin(X(0, i));
	}

	/* shogun representation */
	CDenseFeatures<float64_t>* feat_train=new CDenseFeatures<float64_t>(X);
	CDenseFeatures<float64_t>* feat_test=new CDenseFeatures<float64_t>(X_test);
	CRegressionLabels* label_train=new CRegressionLabels(Y);

	/* specity GPR with exact inference */
	float64_t sigma=1;
	float64_t shogun_sigma=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, shogun_sigma);
	CZeroMean* mean=new CZeroMean();
	CGaussianLikelihood* lik=new CGaussianLikelihood();
	lik->set_sigma(1);
	CExactInferenceMethod* inf=new CExactInferenceMethod(kernel, feat_train,
			mean, label_train, lik);
	CGaussianProcessRegression* gpr=new CGaussianProcessRegression(inf,
			feat_train, label_train);

	/* perform inference */
	gpr->set_return_type(CGaussianProcessRegression::GP_RETURN_MEANS);
	CRegressionLabels* predictions=gpr->apply_regression(feat_test);

	/* do some checks against gpml toolbox*/

	/* m =

   0.221198406887592
   0.537437461176145
   0.431605035301329
   0.373048041692408
   0.253688340068952 */
	SGVector<float64_t> prediction_vector=predictions->get_labels();
//	EXPECT_LE(CMath::abs(prediction_vector[0]-0.221198406887592), 10E-15);
//	EXPECT_LE(CMath::abs(prediction_vector[1]-0.537437461176145), 10E-15);
//	EXPECT_LE(CMath::abs(prediction_vector[2]-0.431605035301329), 10E-15);
//	EXPECT_LE(CMath::abs(prediction_vector[2]-0.373048041692408), 10E-15);
//	EXPECT_LE(CMath::abs(prediction_vector[2]-0.253688340068952), 10E-15);


	SG_UNREF(predictions);
	SG_UNREF(gpr);
}

TEST(GaussianProcessRegression, get_mean_vector)
{
	/* create some easy regression data: 1d noisy sine wave */
	index_t n=3;

	SGMatrix<float64_t> X(1, n);
	SGMatrix<float64_t> X_test(1, n);
	SGVector<float64_t> Y(n);

	X[0]=0;
	X[1]=1.1;
	X[2]=2.2;

	X_test[0]=0.3;
	X_test[1]=1.3;
	X_test[2]=2.5;

	for (index_t i=0; i<n; ++i)
	{
		Y[i]=CMath::sin(X(0, i));
	}

	/* shogun representation */
	CDenseFeatures<float64_t>* feat_train=new CDenseFeatures<float64_t>(X);
	CDenseFeatures<float64_t>* feat_test=new CDenseFeatures<float64_t>(X_test);
	CRegressionLabels* label_train=new CRegressionLabels(Y);

	/* specity GPR with exact inference */
	float64_t sigma=1;
	float64_t shogun_sigma=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, shogun_sigma);
	CZeroMean* mean=new CZeroMean();
	CGaussianLikelihood* lik=new CGaussianLikelihood();
	lik->set_sigma(1);
	CExactInferenceMethod* inf=new CExactInferenceMethod(kernel, feat_train,
			mean, label_train, lik);
	CGaussianProcessRegression* gpr=new CGaussianProcessRegression(inf,
			feat_train, label_train);

	/* perform inference */
	gpr->set_return_type(CGaussianProcessRegression::GP_RETURN_MEANS);
	CRegressionLabels* predictions=gpr->apply_regression(feat_test);

	/* do some checks against gpml toolbox*/

	/* m =

   0.221198406887592
   0.537437461176145
   0.431605035301329 */
	SGVector<float64_t> mean_vector=gpr->get_mean_vector();
	EXPECT_LE(CMath::abs(mean_vector[0]-0.221198406887592), 10E-15);
	EXPECT_LE(CMath::abs(mean_vector[1]-0.537437461176145), 10E-15);
	EXPECT_LE(CMath::abs(mean_vector[2]-0.431605035301329), 10E-15);

	SG_UNREF(predictions);
	SG_UNREF(gpr);
}

TEST(GaussianProcessRegression, get_covariance_vector)
{
	/* create some easy regression data: 1d noisy sine wave */
	index_t n=3;

	SGMatrix<float64_t> X(1, n);
	SGMatrix<float64_t> X_test(1, n);
	SGVector<float64_t> Y(n);

	X[0]=0;
	X[1]=1.1;
	X[2]=2.2;

	X_test[0]=0.3;
	X_test[1]=1.3;
	X_test[2]=2.5;

	for (index_t i=0; i<n; ++i)
	{
		Y[i]=CMath::sin(X(0, i));
	}

	/* shogun representation */
	CDenseFeatures<float64_t>* feat_train=new CDenseFeatures<float64_t>(X);
	CDenseFeatures<float64_t>* feat_test=new CDenseFeatures<float64_t>(X_test);
	CRegressionLabels* label_train=new CRegressionLabels(Y);

	/* specity GPR with exact inference */
	float64_t sigma=1;
	float64_t shogun_sigma=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, shogun_sigma);
	CZeroMean* mean=new CZeroMean();
	CGaussianLikelihood* lik=new CGaussianLikelihood();
	lik->set_sigma(1);
	CExactInferenceMethod* inf=new CExactInferenceMethod(kernel, feat_train,
			mean, label_train, lik);
	CGaussianProcessRegression* gpr=new CGaussianProcessRegression(inf,
			feat_train, label_train);

	/* perform inference */
	gpr->set_return_type(CGaussianProcessRegression::GP_RETURN_MEANS);
	CRegressionLabels* predictions=gpr->apply_regression(feat_test);

	/* do some checks against gpml toolbox*/

	/* s2 =

   1.426104216614624
   1.416896787316447
   1.535464779087576 */
	SGVector<float64_t> covariance_vector=gpr->get_covariance_vector();
	EXPECT_LE(CMath::abs(covariance_vector[0]-1.426104216614624), 10E-15);
	EXPECT_LE(CMath::abs(covariance_vector[1]-1.416896787316447), 10E-15);
	EXPECT_LE(CMath::abs(covariance_vector[2]-1.535464779087576), 10E-15);

	SG_UNREF(predictions);
	SG_UNREF(gpr);
}

#endif
