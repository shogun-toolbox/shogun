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

#include <shogun/lib/config.h>

#if defined(HAVE_EIGEN3) && defined(HAVE_LINALG_LIB)

#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/machine/gp/GaussianARDFITCKernel.h>
#include <shogun/machine/gp/FITCInferenceMethod.h>
#include <shogun/machine/gp/ConstMean.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(SingleFITCInferenceBase,set_kernel)
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
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CDenseFeatures<int32_t>* inducing_features_train=new CDenseFeatures<int32_t>(lat_feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	float64_t ell=1.0;
	CLinearARDKernel* kernel=new CGaussianARDFITCKernel(10, 2*ell*ell);
	float64_t weight1=3.0;
	float64_t weight2=2.0;
	SGVector<float64_t> weights(2);
	weights[0]=1.0/weight1;
	weights[1]=1.0/weight2;
	kernel->set_vector_weights(weights);

	float64_t mean_weight=2.0;
	CConstMean* mean=new CConstMean(mean_weight);

	// Gaussian likelihood with sigma = 0.1
	float64_t sigma=0.1;
	CGaussianLikelihood* lik=new CGaussianLikelihood(sigma);

	// specify GP regression with FITC inference
	CFITCInferenceMethod* inf=new CFITCInferenceMethod(kernel, features_train,
		mean, labels_train, lik, inducing_features_train);

	float64_t ind_noise=1e-6*CMath::sq(sigma);
	inf->set_inducing_noise(ind_noise);

	float64_t scale=3.0;
	inf->set_scale(scale);

	// build parameter dictionary
	CMap<TParameter*, CSGObject*>* parameter_dictionary=new CMap<TParameter*, CSGObject*>();
	inf->build_gradient_parameter_dictionary(parameter_dictionary);

	// compute derivatives wrt parameters
	CMap<TParameter*, SGVector<float64_t> >* gradient=
		inf->get_negative_log_marginal_likelihood_derivatives(parameter_dictionary);

	// get parameters to compute derivatives
	//TParameter* width_param=kernel->m_gradient_parameters->get_parameter("width");
	TParameter* scale_param=inf->m_gradient_parameters->get_parameter("log_scale");
	TParameter* sigma_param=lik->m_gradient_parameters->get_parameter("sigma");
	TParameter* noise_param=inf->m_gradient_parameters->get_parameter("inducing_noise");
	TParameter* weights_param=kernel->m_gradient_parameters->get_parameter("weights");
	TParameter* mean_param=mean->m_gradient_parameters->get_parameter("mean");

	float64_t dnlZ_sf2=(gradient->get_element(scale_param))[0];
	float64_t dnlZ_lik=(gradient->get_element(sigma_param))[0];
	float64_t dnlZ_noise=(gradient->get_element(noise_param))[0];
	float64_t dnlZ_mean=(gradient->get_element(mean_param))[0];

	float64_t dnlz_weight1=(-1.0/weight1)*(gradient->get_element(weights_param))[0];
	float64_t dnlz_weight2=(-1.0/weight2)*(gradient->get_element(weights_param))[1];


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

	TParameter* lat_param=inf->m_gradient_parameters->get_parameter("inducing_features");
	SGVector<float64_t> tmp=gradient->get_element(lat_param);
	SGMatrix<float64_t> deriv_lat(tmp.vector, dim, m, false);

	EXPECT_NEAR(deriv_lat(0,0), -0.005894683010959, 1E-10);
	EXPECT_NEAR(deriv_lat(1,0), -0.089218129675339, 1E-10);

	EXPECT_NEAR(deriv_lat(0,1), 0.187319402698941, 1E-10);
	EXPECT_NEAR(deriv_lat(1,1), 0.276183437603501, 1E-10);

	EXPECT_NEAR(deriv_lat(0,2), 0.101340718980114, 1E-10);
	EXPECT_NEAR(deriv_lat(1,2), 0.127863406431433, 1E-10);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);


	CKernel* kernel2=new CGaussianKernel(10, 2*ell*ell);
	inf->set_kernel(kernel2);

	// build parameter dictionary
	CMap<TParameter*, CSGObject*>* parameter_dictionary2=new CMap<TParameter*, CSGObject*>();
	inf->build_gradient_parameter_dictionary(parameter_dictionary2);

	// compute derivatives wrt parameters
	CMap<TParameter*, SGVector<float64_t> >* gradient2=
		inf->get_negative_log_marginal_likelihood_derivatives(parameter_dictionary2);

	TParameter* lat_param2=inf->m_gradient_parameters->get_parameter("inducing_features");
	SGVector<float64_t> tmp2=gradient2->get_element(lat_param2);
	SGMatrix<float64_t> deriv_lat2(tmp2.vector, dim, m, false);

	//Since CGaussianKernel does not fully support FITC inference, the derivatives are all zeros.
	EXPECT_NEAR(deriv_lat2(0,0), 0, 1E-10);
	EXPECT_NEAR(deriv_lat2(1,0), 0, 1E-10);

	EXPECT_NEAR(deriv_lat2(0,1), 0, 1E-10);
	EXPECT_NEAR(deriv_lat2(1,1), 0, 1E-10);

	EXPECT_NEAR(deriv_lat2(0,2), 0, 1E-10);
	EXPECT_NEAR(deriv_lat2(1,2), 0, 1E-10);

	SG_UNREF(gradient2);
	SG_UNREF(parameter_dictionary2);

	SG_UNREF(inf);
}
#endif /* HAVE_EIGEN3 && HAVE_LINALG_LIB */
