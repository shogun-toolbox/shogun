/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2015 Wu Lin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
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
 *
 */

#include <gtest/gtest.h>
#include <shogun/lib/config.h>

#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/machine/gp/GaussianARDSparseKernel.h>
#include <shogun/machine/gp/FITCInferenceMethod.h>
#include <shogun/machine/gp/ConstMean.h>
#include <shogun/machine/gp/GaussianLikelihood.h>

using namespace shogun;

TEST(SingleFITCInference,set_kernel)
{
	// create some easy regression data with inducing features:
	// y approximately equals to x^sin(x)
	index_t n=6;
	index_t dim=2;
	index_t m=3;

	SGMatrix<float64_t> feat_train(dim, n);
	SGMatrix<int32_t> lat_feat_train(dim, m);
	SGVector<float64_t> lab_train(n);

	feat_train(0,0)=0.81263;
	feat_train(0,1)=0.99976;
	feat_train(0,2)=1.17037;
	feat_train(0,3)=1.51752;
	feat_train(0,4)=1.57765;
	feat_train(0,5)=3.89440;

	feat_train(1,0)=0.5;
	feat_train(1,1)=0.4576;
	feat_train(1,2)=5.17637;
	feat_train(1,3)=2.56752;
	feat_train(1,4)=4.57765;
	feat_train(1,5)=2.89440;

	lat_feat_train(0,0)=1;
	lat_feat_train(0,1)=3;
	lat_feat_train(0,2)=4;

	lat_feat_train(1,0)=3;
	lat_feat_train(1,1)=2;
	lat_feat_train(1,2)=5;

	lab_train[0]=0.46015;
	lab_train[1]=0.69979;
	lab_train[2]=2.15589;
	lab_train[3]=1.51672;
	lab_train[4]=3.59764;
	lab_train[5]=2.39475;

	// shogun representation of features and labels
	auto features_train=std::make_shared<DenseFeatures<float64_t>>(feat_train);
	auto inducing_features_train=std::make_shared<DenseFeatures<int32_t>>(lat_feat_train);
	auto labels_train=std::make_shared<RegressionLabels>(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	auto kernel=std::make_shared<GaussianARDSparseKernel>(10);
	float64_t weight1=3.0;
	float64_t weight2=2.0;
	SGVector<float64_t> weights(2);
	weights[0]=1.0/weight1;
	weights[1]=1.0/weight2;
	kernel->set_vector_weights(weights);

	float64_t mean_weight=2.0;
	auto mean=std::make_shared<ConstMean>(mean_weight);

	// Gaussian likelihood with sigma = 0.1
	float64_t sigma=0.1;
	auto lik=std::make_shared<GaussianLikelihood>(sigma);

	// specify GP regression with FITC inference
	auto inf=std::make_shared<FITCInferenceMethod>(kernel, features_train,
		mean, labels_train, lik, inducing_features_train);

	float64_t ind_noise=1e-6*Math::sq(sigma);
	inf->set_inducing_noise(ind_noise);

	float64_t scale=3.0;
	inf->set_scale(scale);

	// build parameter dictionary
	std::map<std::pair<std::string, std::shared_ptr<const AnyParameter>>, std::shared_ptr<SGObject>> parameter_dictionary;
	inf->build_gradient_parameter_dictionary(parameter_dictionary);

	// compute derivatives wrt parameters
	auto gradient=
		inf->get_negative_log_marginal_likelihood_derivatives(parameter_dictionary);

	// get parameters to compute derivatives
	//TParameter* width_param=kernel->m_gradient_parameters->get_parameter("width");

	float64_t dnlZ_sf2=gradient["log_scale"][0];
	float64_t dnlZ_lik=gradient["log_sigma"][0];
	float64_t dnlZ_noise=gradient["log_inducing_noise"][0];
	float64_t dnlZ_mean=gradient["mean"][0];

	float64_t dnlz_weight1=-gradient["log_weights"][0];
	float64_t dnlz_weight2=-gradient["log_weights"][1];

	dnlZ_lik+=dnlZ_noise;

	// comparison of partial derivatives of negative log marginal likelihood
	// with result from GPML 3.5 package:

	// lik = 0.011674937676697
	// cov =
	// -1.198447375458836
	// -1.690346553000929
	// 4.807315281470778
	// xu =
	// -0.005894683010959   0.187319402698941   0.101340718980114
	// -0.089218129675339   0.276183437603501   0.127863406431433
	// mean = 0.095698957468910

	EXPECT_NEAR(dnlZ_lik, 0.011674937676697, 1E-10);
	EXPECT_NEAR(dnlZ_mean, 0.095698957468910, 1E-10);
	EXPECT_NEAR(dnlZ_sf2, 4.807315281470778, 1E-10);
	EXPECT_NEAR(dnlz_weight1,-1.198447375458836, 1E-10);
	EXPECT_NEAR(dnlz_weight2, -1.690346553000929, 1E-10);

	SGVector<float64_t> tmp=gradient["inducing_features"];
	SGMatrix<float64_t> deriv_lat(tmp.vector, dim, m, false);

	EXPECT_NEAR(deriv_lat(0,0), -0.005894683010959, 1E-10);
	EXPECT_NEAR(deriv_lat(1,0), -0.089218129675339, 1E-10);

	EXPECT_NEAR(deriv_lat(0,1), 0.187319402698941, 1E-10);
	EXPECT_NEAR(deriv_lat(1,1), 0.276183437603501, 1E-10);

	EXPECT_NEAR(deriv_lat(0,2), 0.101340718980114, 1E-10);
	EXPECT_NEAR(deriv_lat(1,2), 0.127863406431433, 1E-10);

	auto kernel2=std::make_shared<GaussianKernel>(10, 2.0);
	inf->set_kernel(kernel2);

	// build parameter dictionary
	std::map<std::pair<std::string, std::shared_ptr<const AnyParameter>>, std::shared_ptr<SGObject>> parameter_dictionary2;
	inf->build_gradient_parameter_dictionary(parameter_dictionary2);

	// compute derivatives wrt parameters
	auto gradient2=
		inf->get_negative_log_marginal_likelihood_derivatives(parameter_dictionary2);

	SGVector<float64_t> tmp2=gradient2["inducing_features"];
	SGMatrix<float64_t> deriv_lat2(tmp2.vector, dim, m, false);

	//Since GaussianKernel does not fully support FITC inference, the derivatives are all zeros.
	EXPECT_NEAR(deriv_lat2(0,0), 0, 1E-10);
	EXPECT_NEAR(deriv_lat2(1,0), 0, 1E-10);

	EXPECT_NEAR(deriv_lat2(0,1), 0, 1E-10);
	EXPECT_NEAR(deriv_lat2(1,1), 0, 1E-10);

	EXPECT_NEAR(deriv_lat2(0,2), 0, 1E-10);
	EXPECT_NEAR(deriv_lat2(1,2), 0, 1E-10);
}
