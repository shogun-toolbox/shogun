/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Roman Votyakov, Wu Lin, Pan Deng, Akash Shivram
 */

#include <gtest/gtest.h>
#include <shogun/lib/config.h>

#include <shogun/labels/RegressionLabels.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/machine/gp/SingleLaplaceInferenceMethod.h>
#include <shogun/machine/gp/EPInferenceMethod.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/LogitLikelihood.h>
#include <shogun/machine/gp/ExactInferenceMethod.h>
#include <shogun/machine/gp/GaussianLikelihood.h>

using namespace shogun;

#ifdef USE_GPL_SHOGUN
TEST(InferenceMethod,get_marginal_likelihood_estimate_logit_laplace)
{
	index_t n=2;
	index_t d=1;

	SGMatrix<float64_t> feat_train(d, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0,0)=1;
	feat_train(0,1)=-1;
	lab_train[0]=1;
	lab_train[1]=-1;

	auto features_train=std::make_shared<DenseFeatures<float64_t>>(feat_train);
	auto labels_train=std::make_shared<BinaryLabels>(lab_train);

	auto kernel=std::make_shared<GaussianKernel>(10, 8);
	auto mean=std::make_shared<ZeroMean>();
	auto likelihood=std::make_shared<LogitLikelihood>();
	auto inf=std::make_shared<SingleLaplaceInferenceMethod>(kernel,
			features_train,	mean, labels_train, likelihood);
	inf->set_scale(2.0);

	/* sample estimate and compare against a number from my python
	 * implementation, and also against the approximate marginal
	 * likelihood. Since this is random, use low accuracy. */
	float64_t sample=inf->get_marginal_likelihood_estimate(100000);
	EXPECT_NEAR(sample, -1.67990517588, 0.3);
	EXPECT_NEAR(sample, -inf->get_negative_log_marginal_likelihood(), 0.3);


}

TEST(InferenceMethod,get_marginal_likelihood_estimate_logit_ep)
{
	index_t n=2;
	index_t d=1;

	SGMatrix<float64_t> feat_train(d, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0,0)=1;
	feat_train(0,1)=-1;
	lab_train[0]=1;
	lab_train[1]=-1;

	auto features_train=std::make_shared<DenseFeatures<float64_t>>(feat_train);
	auto labels_train=std::make_shared<BinaryLabels>(lab_train);

	auto kernel=std::make_shared<GaussianKernel>(10, 8);
	auto mean=std::make_shared<ZeroMean>();
	auto likelihood=std::make_shared<LogitLikelihood>();
	auto inf=std::make_shared<EPInferenceMethod>(kernel, features_train, mean,
			labels_train, likelihood);
	inf->set_scale(2.0);

	// sample estimate and compare against the approximate marginal
	// likelihood. Since this is random, use low accuracy.
	float64_t sample=inf->get_marginal_likelihood_estimate(100000);
	EXPECT_NEAR(sample, -inf->get_negative_log_marginal_likelihood(), 1E-2);


}
#endif //USE_GPL_SHOGUN

TEST(InferenceMethod, compute_gradient)
{
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
		Y[i] = std::sin(X(0, i));
	}

	auto feat_train=std::make_shared<DenseFeatures<float64_t>>(X);
	auto label_train=std::make_shared<RegressionLabels>(Y);

	float64_t sigma=1;
	float64_t shogun_sigma=sigma*sigma*2;
	auto kernel=std::make_shared<GaussianKernel>(10, shogun_sigma);
	auto mean=std::make_shared<ZeroMean>();
	auto lik=std::make_shared<GaussianLikelihood>();
	lik->set_sigma(1);
	auto inf=std::make_shared<ExactInferenceMethod>(kernel, feat_train,
			mean, label_train, lik);

	SGMatrix<float64_t> L=inf->get_cholesky();
	uint32_t hash1=inf->m_hash;

	L=inf->get_cholesky();
	uint32_t hash2=inf->m_hash;
	EXPECT_TRUE(hash1==hash2);

	SGMatrix<float64_t> Sigma=inf->get_posterior_covariance();
	uint32_t hash3=inf->m_hash;
	EXPECT_TRUE(hash2!=hash3);

	Sigma=inf->get_posterior_covariance();
	uint32_t hash4=inf->m_hash;
	EXPECT_TRUE(hash3==hash4);

	L=inf->get_cholesky();
	uint32_t hash5=inf->m_hash;
	EXPECT_TRUE(hash4==hash5);

	Sigma=inf->get_posterior_covariance();
	uint32_t hash6=inf->m_hash;
	EXPECT_TRUE(hash5==hash6);

	Sigma=inf->get_posterior_covariance();
	uint32_t hash7=inf->m_hash;
	EXPECT_TRUE(hash6==hash7);


}
