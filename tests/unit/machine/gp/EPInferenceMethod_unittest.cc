/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Roman Votyakov, Wu Lin, Pan Deng
 */

#include <gtest/gtest.h>
#include <shogun/lib/config.h>

#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/machine/gp/EPInferenceMethod.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/ProbitLikelihood.h>

using namespace shogun;

TEST(EPInferenceMethod,get_cholesky_probit_likelihood)
{
	// create some easy random classification data
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0,0)=-1.07932;
	feat_train(0,1)=1.15768;
	feat_train(0,2)=3.26631;
	feat_train(0,3)=1.79009;
	feat_train(0,4)=-3.66051;

	feat_train(1,0)=-1.83544;
	feat_train(1,1)=2.91702;
	feat_train(1,2)=-3.85663;
	feat_train(1,3)=0.11949;
	feat_train(1,4)=1.75159;

	lab_train[0]=-1.0;
	lab_train[1]=1.0;
	lab_train[2]=1.0;
	lab_train[3]=1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	auto features_train=std::make_shared<DenseFeatures<float64_t>>(feat_train);
	auto labels_train=std::make_shared<BinaryLabels>(lab_train);

	// choose Gaussian kernel with width = 2*2^2 and zero mean function
	auto kernel=std::make_shared<GaussianKernel>(10, 8.0);
	auto mean=std::make_shared<ZeroMean>();

	// probit likelihood
	auto likelihood=std::make_shared<ProbitLikelihood>();

	// specify GP classification with EP inference and kernel scale=1.5
	auto inf=std::make_shared<EPInferenceMethod>(kernel,
		features_train,	mean, labels_train, likelihood);
	inf->set_scale(1.5);

	// comparison of cholesky with result from GPML package
	SGMatrix<float64_t> L=inf->get_cholesky();

	EXPECT_NEAR(L(0,0), 1.358253004928362, 1E-3);
	EXPECT_NEAR(L(0,1), 0.018316522108192, 1E-3);
	EXPECT_NEAR(L(0,2), 0.033812347702551, 1E-3);
	EXPECT_NEAR(L(0,3), 0.130014750307937, 1E-3);
	EXPECT_NEAR(L(0,4), 0.051980118062897, 1E-3);

	EXPECT_NEAR(L(1,0), 0.000000000000000, 1E-3);
	EXPECT_NEAR(L(1,1), 1.313315812622500, 1E-3);
	EXPECT_NEAR(L(1,2), 0.000588353671333, 1E-3);
	EXPECT_NEAR(L(1,3), 0.199232686436273, 1E-3);
	EXPECT_NEAR(L(1,4), 0.025787680602556, 1E-3);

	EXPECT_NEAR(L(2,0), 0.000000000000000, 1E-3);
	EXPECT_NEAR(L(2,1), 0.000000000000000, 1E-3);
	EXPECT_NEAR(L(2,2), 1.333160257955935, 1E-3);
	EXPECT_NEAR(L(2,3), 0.057177746824419, 1E-3);
	EXPECT_NEAR(L(2,4), -0.001301272388376, 1E-3);

	EXPECT_NEAR(L(3,0), 0.000000000000000, 1E-3);
	EXPECT_NEAR(L(3,1), 0.000000000000000, 1E-3);
	EXPECT_NEAR(L(3,2), 0.000000000000000, 1E-3);
	EXPECT_NEAR(L(3,3), 1.300708604766544, 1E-3);
	EXPECT_NEAR(L(3,4), 0.001192632066695, 1E-3);

	EXPECT_NEAR(L(4,0), 0.000000000000000, 1E-3);
	EXPECT_NEAR(L(4,1), 0.000000000000000, 1E-3);
	EXPECT_NEAR(L(4,2), 0.000000000000000, 1E-3);
	EXPECT_NEAR(L(4,3), 0.000000000000000, 1E-3);
	EXPECT_NEAR(L(4,4), 1.332317592760179, 1E-3);

	// clean up

}

TEST(EPInferenceMethod,get_negative_marginal_likelihood_probit_likelihood)
{
	// create some easy random classification data
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0,0)=-1.07932;
	feat_train(0,1)=1.15768;
	feat_train(0,2)=3.26631;
	feat_train(0,3)=1.79009;
	feat_train(0,4)=-3.66051;

	feat_train(1,0)=-1.83544;
	feat_train(1,1)=2.91702;
	feat_train(1,2)=-3.85663;
	feat_train(1,3)=0.11949;
	feat_train(1,4)=1.75159;

	lab_train[0]=-1.0;
	lab_train[1]=1.0;
	lab_train[2]=1.0;
	lab_train[3]=1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	auto features_train=std::make_shared<DenseFeatures<float64_t>>(feat_train);
	auto labels_train=std::make_shared<BinaryLabels>(lab_train);

	// choose Gaussian kernel with width = 2*2^2 and zero mean function
	auto kernel=std::make_shared<GaussianKernel>(10, 8.0);
	auto mean=std::make_shared<ZeroMean>();

	// probit likelihood
	auto likelihood=std::make_shared<ProbitLikelihood>();

	// specify GP classification with EP inference and kernel scale=1.5
	auto inf=std::make_shared<EPInferenceMethod>(kernel,
		features_train,	mean, labels_train, likelihood);
	inf->set_scale(1.5);

	// comparison of negative marginal likelihood with result from GPML package
	float64_t nlZ=inf->get_negative_log_marginal_likelihood();

	EXPECT_NEAR(nlZ, 3.38359489001561, 1E-3);

	// clean up

}

TEST(EPInferenceMethod,get_alpha_probit_likelihood)
{
	// create some easy random classification data
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0,0)=-1.07932;
	feat_train(0,1)=1.15768;
	feat_train(0,2)=3.26631;
	feat_train(0,3)=1.79009;
	feat_train(0,4)=-3.66051;

	feat_train(1,0)=-1.83544;
	feat_train(1,1)=2.91702;
	feat_train(1,2)=-3.85663;
	feat_train(1,3)=0.11949;
	feat_train(1,4)=1.75159;

	lab_train[0]=-1.0;
	lab_train[1]=1.0;
	lab_train[2]=1.0;
	lab_train[3]=1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	auto features_train=std::make_shared<DenseFeatures<float64_t>>(feat_train);
	auto labels_train=std::make_shared<BinaryLabels>(lab_train);

	// choose Gaussian kernel with width = 2*2^2 and zero mean function
	auto kernel=std::make_shared<GaussianKernel>(10, 8.0);
	auto mean=std::make_shared<ZeroMean>();

	// probit likelihood
	auto likelihood=std::make_shared<ProbitLikelihood>();

	// specify GP classification with EP inference and kernel scale=1.5
	auto inf=std::make_shared<EPInferenceMethod>(kernel,
		features_train,	mean, labels_train, likelihood);
	inf->set_scale(1.5);

	// comparison of alpha with result from GPML package
	SGVector<float64_t> alpha=inf->get_alpha();
	EXPECT_NEAR(alpha[0], -0.481804252788557, 1E-3);
	EXPECT_NEAR(alpha[1], 0.392192885549848, 1E-3);
	EXPECT_NEAR(alpha[2], 0.435105219728697, 1E-3);
	EXPECT_NEAR(alpha[3], 0.407811602073545, 1E-3);
	EXPECT_NEAR(alpha[4], -0.435104577247077, 1E-3);

	// clean up

}

TEST(EPInferenceMethod,get_marginal_likelihood_derivatives_probit_likelihood)
{
	// create some easy random classification data
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0,0)=-1.07932;
	feat_train(0,1)=1.15768;
	feat_train(0,2)=3.26631;
	feat_train(0,3)=1.79009;
	feat_train(0,4)=-3.66051;

	feat_train(1,0)=-1.83544;
	feat_train(1,1)=2.91702;
	feat_train(1,2)=-3.85663;
	feat_train(1,3)=0.11949;
	feat_train(1,4)=1.75159;

	lab_train[0]=-1.0;
	lab_train[1]=1.0;
	lab_train[2]=1.0;
	lab_train[3]=1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	auto features_train=std::make_shared<DenseFeatures<float64_t>>(feat_train);
	auto labels_train=std::make_shared<BinaryLabels>(lab_train);

	float64_t ell=2.0;

	// choose Gaussian kernel with width = 2*2^2 and zero mean function
	auto kernel=std::make_shared<GaussianKernel>(10, 2*Math::sq(ell));
	auto mean=std::make_shared<ZeroMean>();

	// probit likelihood
	auto likelihood=std::make_shared<ProbitLikelihood>();

	// specify GP classification with EP inference and kernel scale=1.5
	auto inf=std::make_shared<EPInferenceMethod>(kernel, features_train, mean,
			labels_train, likelihood);
	inf->set_scale(1.5);

	// build parameter dictionary
	auto parameter_dictionary=std::make_shared<CMap<TParameter*, SGObject*>>();
	inf->build_gradient_parameter_dictionary(parameter_dictionary);

	// compute derivatives wrt parameters
	auto gradient=
		inf->get_negative_log_marginal_likelihood_derivatives(parameter_dictionary);

	// get parameters to compute derivatives
	TParameter* width_param=kernel->m_gradient_parameters->get_parameter("log_width");
	TParameter* scale_param=inf->m_gradient_parameters->get_parameter("log_scale");

	float64_t dnlZ_ell=(gradient->get_element(width_param))[0];
	float64_t dnlZ_sf2=(gradient->get_element(scale_param))[0];

	// comparison of partial derivatives of negative marginal likelihood with
	// result from GPML package:
	EXPECT_NEAR(dnlZ_ell, -0.0551896689012401, 1E-3);
	EXPECT_NEAR(dnlZ_sf2, -0.0535698533526804, 1E-3);

	// clean up



}

TEST(EPInferenceMethod, get_posterior_mean_probit_likelihood)
{
	// create some easy random classification data
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0,0)=-1.07932;
	feat_train(0,1)=1.15768;
	feat_train(0,2)=3.26631;
	feat_train(0,3)=1.79009;
	feat_train(0,4)=-3.66051;

	feat_train(1,0)=-1.83544;
	feat_train(1,1)=2.91702;
	feat_train(1,2)=-3.85663;
	feat_train(1,3)=0.11949;
	feat_train(1,4)=1.75159;

	lab_train[0]=-1.0;
	lab_train[1]=1.0;
	lab_train[2]=1.0;
	lab_train[3]=1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	auto features_train=std::make_shared<DenseFeatures<float64_t>>(feat_train);
	auto labels_train=std::make_shared<BinaryLabels>(lab_train);

	// choose Gaussian kernel with width = 2*2^2 and zero mean function
	auto kernel=std::make_shared<GaussianKernel>(10, 8.0);
	auto mean=std::make_shared<ZeroMean>();

	// probit likelihood
	auto likelihood=std::make_shared<ProbitLikelihood>();

	// specify GP classification with EP inference and kernel scale=1.5
	auto inf=std::make_shared<EPInferenceMethod>(kernel,
		features_train,	mean, labels_train, likelihood);
	inf->set_scale(1.5);

	// comparison of posterior approximation mean with result from GPML package
	SGVector<float64_t> mu=inf->get_posterior_mean();

	EXPECT_NEAR(mu[0], -0.882471450365118, 1E-3);
	EXPECT_NEAR(mu[1], 1.132570041978009, 1E-3);
	EXPECT_NEAR(mu[2], 1.016031341665029, 1E-3);
	EXPECT_NEAR(mu[3], 1.079152436021026, 1E-3);
	EXPECT_NEAR(mu[4], -1.016378075891627, 1E-3);

	// clean up

}

TEST(EPInferenceMethod, get_posterior_covariance_probit_likelihood)
{
	// create some easy random classification data
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0,0)=-1.07932;
	feat_train(0,1)=1.15768;
	feat_train(0,2)=3.26631;
	feat_train(0,3)=1.79009;
	feat_train(0,4)=-3.66051;

	feat_train(1,0)=-1.83544;
	feat_train(1,1)=2.91702;
	feat_train(1,2)=-3.85663;
	feat_train(1,3)=0.11949;
	feat_train(1,4)=1.75159;

	lab_train[0]=-1.0;
	lab_train[1]=1.0;
	lab_train[2]=1.0;
	lab_train[3]=1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	auto features_train=std::make_shared<DenseFeatures<float64_t>>(feat_train);
	auto labels_train=std::make_shared<BinaryLabels>(lab_train);

	// choose Gaussian kernel with width = 2*2^2 and zero mean function
	auto kernel=std::make_shared<GaussianKernel>(10, 8.0);
	auto mean=std::make_shared<ZeroMean>();

	// probit likelihood
	auto likelihood=std::make_shared<ProbitLikelihood>();

	// specify GP classification with EP inference and kernel scale=1.5
	auto inf=std::make_shared<EPInferenceMethod>(kernel,
		features_train,	mean, labels_train, likelihood);
	inf->set_scale(1.5);

	// comparison of posterior approximation covariance with result from GPML
	// package
	SGMatrix<float64_t> Sigma=inf->get_posterior_covariance();

	EXPECT_NEAR(Sigma(0,0), 1.20274103263760379, 1E-3);
	EXPECT_NEAR(Sigma(0,1), -0.00260850144375850, 1E-3);
	EXPECT_NEAR(Sigma(0,2), 0.03239746041777301, 1E-3);
	EXPECT_NEAR(Sigma(0,3), 0.15449141486321272, 1E-3);
	EXPECT_NEAR(Sigma(0,4), 0.05930784879253234, 1E-3);

	EXPECT_NEAR(Sigma(1,0), -0.00260850144375854, 1E-3);
	EXPECT_NEAR(Sigma(1,1), 1.26103532435204135, 1E-3);
	EXPECT_NEAR(Sigma(1,2), -0.01072708038072782, 1E-3);
	EXPECT_NEAR(Sigma(1,3), 0.27319700541557035, 1E-3);
	EXPECT_NEAR(Sigma(1,4), 0.03289357125150720, 1E-3);

	EXPECT_NEAR(Sigma(2,0), 0.03239746041777301, 1E-3);
	EXPECT_NEAR(Sigma(2,1), -0.01072708038072782, 1E-3);
	EXPECT_NEAR(Sigma(2,2), 1.26094966657088081, 1E-3);
	EXPECT_NEAR(Sigma(2,3), 0.07456464702006255, 1E-3);
	EXPECT_NEAR(Sigma(2,4), -0.00165339284404121, 1E-3);

	EXPECT_NEAR(Sigma(3,0), 0.15449141486321255, 1E-3);
	EXPECT_NEAR(Sigma(3,1), 0.27319700541557035, 1E-3);
	EXPECT_NEAR(Sigma(3,2), 0.07456464702006255, 1E-3);
	EXPECT_NEAR(Sigma(3,3), 1.22399410182504154, 1E-3);
	EXPECT_NEAR(Sigma(3,4), 0.00151934275843193, 1E-3);

	EXPECT_NEAR(Sigma(4,0), 0.05930784879253234, 1E-3);
	EXPECT_NEAR(Sigma(4,1), 0.03289357125150720, 1E-3);
	EXPECT_NEAR(Sigma(4,2), -0.00165339284404121, 1E-3);
	EXPECT_NEAR(Sigma(4,3), 0.00151934275843193, 1E-3);
	EXPECT_NEAR(Sigma(4,4), 1.26206797645117108, 1E-3);

	// clean up

}
