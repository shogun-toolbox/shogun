#include <shogun/lib/config.h>
#ifdef HAVE_EIGEN3

#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/regression/gp/ExactInferenceMethod.h>
#include <shogun/regression/gp/ZeroMean.h>
#include <shogun/regression/gp/GaussianLikelihood.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(ExactInferenceMethod,get_cholesky)
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

	/* do some checks against gpml toolbox*/

	/*
L =

   1.414213562373095   0.386132930109494   0.062877078699608
                   0   1.360478357154224   0.383538270389077
                   0                   0   1.359759121359794
    */
	SGMatrix<float64_t> L=inf->get_cholesky();
	EXPECT_LE(CMath::abs(L(0,0)-1.414213562373095), 10E-15);
	EXPECT_LE(CMath::abs(L(0,1)-0.386132930109494), 10E-15);
	EXPECT_LE(CMath::abs(L(0,2)-0.062877078699608), 10E-15);
	EXPECT_LE(CMath::abs(L(1,0)-0), 10E-15);
	EXPECT_LE(CMath::abs(L(1,1)-1.360478357154224), 10E-15);
	EXPECT_LE(CMath::abs(L(1,2)-0.383538270389077), 10E-15);
	EXPECT_LE(CMath::abs(L(2,0)-0), 10E-15);
	EXPECT_LE(CMath::abs(L(2,1)-0), 10E-15);
	EXPECT_LE(CMath::abs(L(2,2)-1.359759121359794), 10E-15);

	SG_UNREF(inf);
}

TEST(ExactInferenceMethod,get_alpha)
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

	/* do some checks against gpml toolbox*/

	/* alpha =

  -0.121668320184276
   0.396533145765454
   0.301389368713216 */
	SGVector<float64_t> alpha=inf->get_alpha();
	EXPECT_LE(CMath::abs(alpha[0]+0.121668320184276), 10E-15);
	EXPECT_LE(CMath::abs(alpha[1]-0.396533145765454), 10E-15);
	EXPECT_LE(CMath::abs(alpha[2]-0.301389368713216), 10E-15);

	SG_UNREF(inf);
}

TEST(ExactInferenceMethod,get_negative_marginal_likelihood)
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

	/* do some checks against gpml toolbox*/

	/* nlZ =

   4.017065867797999 */
	float64_t nml=inf->get_negative_marginal_likelihood();
	EXPECT_LE(CMath::abs(nml-4.017065867797999), 10E-15);

	SG_UNREF(inf);
}

#endif
