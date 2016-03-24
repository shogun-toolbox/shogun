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

#if defined(HAVE_LINALG_LIB)
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/machine/gp/ConstMean.h>
#include <shogun/machine/gp/GaussianARDSparseKernel.h>
#include <shogun/machine/gp/SingleFITCLaplacianInferenceMethod.h>
#include <shogun/machine/gp/LogitLikelihood.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(SingleFITCLaplacianInferenceMethod,get_cholesky)
{
	index_t n=6;
	index_t dim=2;
	index_t m=3;
	float64_t rel_tolorance=1e-5;
	float64_t abs_tolorance;

	SGMatrix<float64_t> feat_train(dim, n);
	SGMatrix<float64_t> lat_feat_train(dim, m);
	SGVector<float64_t> lab_train(n);

	feat_train(0,0)=-0.81263;
	feat_train(0,1)=-0.99976;
	feat_train(0,2)=1.17037;
	feat_train(0,3)=-1.51752;
	feat_train(0,4)=8.57765;
	feat_train(0,5)=3.89440;

	feat_train(1,0)=-0.5;
	feat_train(1,1)=5.4576;
	feat_train(1,2)=7.17637;
	feat_train(1,3)=-2.56752;
	feat_train(1,4)=4.57765;
	feat_train(1,5)=2.89440;

	lat_feat_train(0,0)=1.00000;
	lat_feat_train(0,1)=23.00000;
	lat_feat_train(0,2)=4.00000;

	lat_feat_train(1,0)=3.00000;
	lat_feat_train(1,1)=2.00000;
	lat_feat_train(1,2)=-5.00000;

	lab_train[0]=1;
	lab_train[1]=-1;
	lab_train[2]=1;
	lab_train[3]=1;
	lab_train[4]=-1;
	lab_train[5]=-1;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(lat_feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2
	CGaussianARDSparseKernel* kernel=new CGaussianARDSparseKernel(10);
	int32_t t_dim=2;
	SGMatrix<float64_t> weights(dim,t_dim);
	//the weights is a lower triangular matrix
	float64_t weight1=0.02;
	float64_t weight2=-0.4;
	float64_t weight3=0;
	float64_t weight4=0.01;
	weights(0,0)=weight1;
	weights(1,0)=weight2;
	weights(0,1)=weight3;
	weights(1,1)=weight4;
	kernel->set_matrix_weights(weights);

	float64_t mean_weight=2.0;
	CConstMean* mean=new CConstMean(mean_weight);

	CLogitLikelihood* lik=new CLogitLikelihood();

	// specify GP with FITC inference
	CSingleFITCLaplacianInferenceMethod* inf=new CSingleFITCLaplacianInferenceMethod(kernel, features_train,
		mean, labels_train, lik, latent_features_train);

	float64_t ind_noise=1e-6;
	inf->set_inducing_noise(ind_noise);

	float64_t scale=4.0;
	inf->set_scale(scale);
	// comparison of posterior cholesky with result from GPML 3.5 package:
	// L =
	//-0.079160062284788  0.043573099312206   0.003733704862708
	//0.043573099312206  -0.062046601915619  -0.006797768274130
	//0.003733704862708  -0.006797768274130  -0.006876868793106

	SGMatrix<float64_t> L=inf->get_cholesky();

	abs_tolorance = CMath::get_abs_tolerance(-0.079160062284788, rel_tolorance);
	EXPECT_NEAR(L(0,0),  -0.079160062284788,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolerance(0.043573099312206, rel_tolorance);
	EXPECT_NEAR(L(0,1),  0.043573099312206,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolerance(0.003733704862708, rel_tolorance);
	EXPECT_NEAR(L(0,2),  0.003733704862708,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolerance(0.043573099312206, rel_tolorance);
	EXPECT_NEAR(L(1,0),  0.043573099312206,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolerance(-0.062046601915619, rel_tolorance);
	EXPECT_NEAR(L(1,1),  -0.062046601915619,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolerance(-0.006797768274130, rel_tolorance);
	EXPECT_NEAR(L(1,2),  -0.006797768274130,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolerance(0.003733704862708, rel_tolorance);
	EXPECT_NEAR(L(2,0),  0.003733704862708,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolerance(-0.006797768274130, rel_tolorance);
	EXPECT_NEAR(L(2,1),  -0.006797768274130,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolerance(-0.006876868793106, rel_tolorance);
	EXPECT_NEAR(L(2,2),  -0.006876868793106,  abs_tolorance);

	SG_UNREF(inf);
}

TEST(SingleFITCLaplacianInferenceMethod,get_alpha)
{
	index_t n=6;
	index_t dim=2;
	index_t m=3;
	float64_t rel_tolorance=1e-5;
	float64_t abs_tolorance;

	SGMatrix<float64_t> feat_train(dim, n);
	SGMatrix<float64_t> lat_feat_train(dim, m);
	SGVector<float64_t> lab_train(n);

	feat_train(0,0)=-0.81263;
	feat_train(0,1)=-0.99976;
	feat_train(0,2)=1.17037;
	feat_train(0,3)=-1.51752;
	feat_train(0,4)=8.57765;
	feat_train(0,5)=3.89440;

	feat_train(1,0)=-0.5;
	feat_train(1,1)=5.4576;
	feat_train(1,2)=7.17637;
	feat_train(1,3)=-2.56752;
	feat_train(1,4)=4.57765;
	feat_train(1,5)=2.89440;

	lat_feat_train(0,0)=1.00000;
	lat_feat_train(0,1)=23.00000;
	lat_feat_train(0,2)=4.00000;

	lat_feat_train(1,0)=3.00000;
	lat_feat_train(1,1)=2.00000;
	lat_feat_train(1,2)=-5.00000;

	lab_train[0]=1;
	lab_train[1]=-1;
	lab_train[2]=1;
	lab_train[3]=1;
	lab_train[4]=-1;
	lab_train[5]=-1;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(lat_feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2
	CGaussianARDSparseKernel* kernel=new CGaussianARDSparseKernel(10);
	int32_t t_dim=2;
	SGMatrix<float64_t> weights(dim,t_dim);
	//the weights is a lower triangular matrix
	float64_t weight1=0.02;
	float64_t weight2=-0.4;
	float64_t weight3=0;
	float64_t weight4=0.01;
	weights(0,0)=weight1;
	weights(1,0)=weight2;
	weights(0,1)=weight3;
	weights(1,1)=weight4;
	kernel->set_matrix_weights(weights);

	float64_t mean_weight=2.0;
	CConstMean* mean=new CConstMean(mean_weight);

	CLogitLikelihood* lik=new CLogitLikelihood();

	// specify GP with FITC inference
	CSingleFITCLaplacianInferenceMethod* inf=new CSingleFITCLaplacianInferenceMethod(kernel, features_train,
		mean, labels_train, lik, latent_features_train);

	float64_t ind_noise=1e-6;
	inf->set_inducing_noise(ind_noise);

	float64_t scale=4.0;
	inf->set_scale(scale);

	// comparison of posterior alpha with result from GPML package:
	// alpha =
	//-0.366893026022809
	//0.174674800097295
	//0.028021450045659
	SGVector<float64_t> alpha=inf->get_alpha();

	abs_tolorance = CMath::get_abs_tolerance(-0.366893026022809, rel_tolorance);
	EXPECT_NEAR(alpha[0],  -0.366893026022809,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolerance(0.174674800097295, rel_tolorance);
	EXPECT_NEAR(alpha[1],  0.174674800097295,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolerance(0.028021450045659, rel_tolorance);
	EXPECT_NEAR(alpha[2],  0.028021450045659,  abs_tolorance);

	SG_UNREF(inf);
}


TEST(SingleFITCLaplacianInferenceMethod,get_negative_log_marginal_likelihood)
{
	index_t n=6;
	index_t dim=2;
	index_t m=3;
	float64_t rel_tolorance=1e-5;
	float64_t abs_tolorance;

	SGMatrix<float64_t> feat_train(dim, n);
	SGMatrix<float64_t> lat_feat_train(dim, m);
	SGVector<float64_t> lab_train(n);

	feat_train(0,0)=-0.81263;
	feat_train(0,1)=-0.99976;
	feat_train(0,2)=1.17037;
	feat_train(0,3)=-1.51752;
	feat_train(0,4)=8.57765;
	feat_train(0,5)=3.89440;

	feat_train(1,0)=-0.5;
	feat_train(1,1)=5.4576;
	feat_train(1,2)=7.17637;
	feat_train(1,3)=-2.56752;
	feat_train(1,4)=4.57765;
	feat_train(1,5)=2.89440;

	lat_feat_train(0,0)=1.00000;
	lat_feat_train(0,1)=23.00000;
	lat_feat_train(0,2)=4.00000;

	lat_feat_train(1,0)=3.00000;
	lat_feat_train(1,1)=2.00000;
	lat_feat_train(1,2)=-5.00000;

	lab_train[0]=1;
	lab_train[1]=-1;
	lab_train[2]=1;
	lab_train[3]=1;
	lab_train[4]=-1;
	lab_train[5]=-1;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(lat_feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianARDSparseKernel* kernel=new CGaussianARDSparseKernel(10);
	int32_t t_dim=2;
	SGMatrix<float64_t> weights(dim,t_dim);
	//the weights is a upper triangular matrix since GPML 3.5 only supports this type
	float64_t weight1=0.02;
	float64_t weight2=-0.4;
	float64_t weight3=0;
	float64_t weight4=0.01;
	weights(0,0)=weight1;
	weights(1,0)=weight2;
	weights(0,1)=weight3;
	weights(1,1)=weight4;
	kernel->set_matrix_weights(weights);

	float64_t mean_weight=2.0;
	CConstMean* mean=new CConstMean(mean_weight);

	CLogitLikelihood* lik=new CLogitLikelihood();

	// specify GP with FITC inference
	CSingleFITCLaplacianInferenceMethod* inf=new CSingleFITCLaplacianInferenceMethod(kernel, features_train,
		mean, labels_train, lik, latent_features_train);

	float64_t ind_noise=1e-6;
	inf->set_inducing_noise(ind_noise);

	float64_t scale=4.0;
	inf->set_scale(scale);

	// comparison of posterior negative marginal likelihood with
	// result from GPML 3.5 package:
	// nlZ =
	// 3.617413494305376
	float64_t nml=inf->get_negative_log_marginal_likelihood();

	abs_tolorance = CMath::get_abs_tolerance(3.617413494305376, rel_tolorance);
	EXPECT_NEAR(nml, 3.617413494305376,  abs_tolorance);

	SG_UNREF(inf);
}

TEST(SingleFITCLaplacianInferenceMethod,get_marginal_likelihood_derivatives)
{
	index_t n=6;
	index_t dim=2;
	index_t m=3;
	float64_t rel_tolorance=1e-5;
	float64_t abs_tolorance;

	SGMatrix<float64_t> feat_train(dim, n);
	SGMatrix<float64_t> lat_feat_train(dim, m);
	SGVector<float64_t> lab_train(n);

	feat_train(0,0)=-0.81263;
	feat_train(0,1)=-0.99976;
	feat_train(0,2)=1.17037;
	feat_train(0,3)=-1.51752;
	feat_train(0,4)=8.57765;
	feat_train(0,5)=3.89440;

	feat_train(1,0)=-0.5;
	feat_train(1,1)=5.4576;
	feat_train(1,2)=7.17637;
	feat_train(1,3)=-2.56752;
	feat_train(1,4)=4.57765;
	feat_train(1,5)=2.89440;

	lat_feat_train(0,0)=1.00000;
	lat_feat_train(0,1)=23.00000;
	lat_feat_train(0,2)=4.00000;

	lat_feat_train(1,0)=3.00000;
	lat_feat_train(1,1)=2.00000;
	lat_feat_train(1,2)=-5.00000;

	lab_train[0]=1;
	lab_train[1]=-1;
	lab_train[2]=1;
	lab_train[3]=1;
	lab_train[4]=-1;
	lab_train[5]=-1;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(lat_feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2
	CGaussianARDSparseKernel* kernel=new CGaussianARDSparseKernel(10);
	int32_t t_dim=2;
	SGMatrix<float64_t> weights(dim,t_dim);
	//the weights is a upper triangular matrix since GPML 3.5 only supports this type
	float64_t weight1=0.02;
	float64_t weight2=-0.4;
	float64_t weight3=0;
	float64_t weight4=0.01;
	weights(0,0)=weight1;
	weights(1,0)=weight2;
	weights(0,1)=weight3;
	weights(1,1)=weight4;
	kernel->set_matrix_weights(weights);

	float64_t mean_weight=2.0;
	CConstMean* mean=new CConstMean(mean_weight);

	CLogitLikelihood* lik=new CLogitLikelihood();

	// specify GP with FITC inference
	CSingleFITCLaplacianInferenceMethod* inf=new CSingleFITCLaplacianInferenceMethod(kernel, features_train,
		mean, labels_train, lik, latent_features_train);

	float64_t ind_noise=1e-6;
	inf->set_inducing_noise(ind_noise);

	float64_t scale=4.0;
	inf->set_scale(scale);

	// build parameter dictionary
	CMap<TParameter*, CSGObject*>* parameter_dictionary=new CMap<TParameter*, CSGObject*>();
	inf->build_gradient_parameter_dictionary(parameter_dictionary);

	// compute derivatives wrt parameters
	CMap<TParameter*, SGVector<float64_t> >* gradient=
		inf->get_negative_log_marginal_likelihood_derivatives(parameter_dictionary);

	// get parameters to compute derivatives
	TParameter* scale_param=inf->m_gradient_parameters->get_parameter("log_scale");
	TParameter* mean_param=mean->m_gradient_parameters->get_parameter("mean");
	TParameter* noise_param=inf->m_gradient_parameters->get_parameter("log_inducing_noise");
	TParameter* weights_param=kernel->m_gradient_parameters->get_parameter("log_weights");


	// result from GPML 3.5 package:
	//dnlz =
	//mean: -0.005787909020849
	//cov: [-0.244187111559869]
	//lik: []
	//noise: 3.423607267495644e-07
	//L2 =
	//0.036402571652033   2.291652389327975
	//0  -0.000362096140397


	float64_t dnlZ_sf2=(gradient->get_element(scale_param))[0];
	float64_t dnlZ_mean=(gradient->get_element(mean_param))[0];
	float64_t dnlZ_noise=(gradient->get_element(noise_param))[0];

	SGVector<float64_t> dnlz_weights_vec=gradient->get_element(weights_param);

	abs_tolorance = CMath::get_abs_tolerance(-0.244187111559869, rel_tolorance);
	EXPECT_NEAR(dnlZ_sf2, -0.244187111559869,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolerance(-0.005787909020849, rel_tolorance);
	EXPECT_NEAR(dnlZ_mean, -0.005787909020849,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolerance(3.423607267495644e-07, rel_tolorance);
	EXPECT_NEAR(dnlZ_noise, 3.423607267495644e-07,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolerance(0.036402571652033, rel_tolorance);
	EXPECT_NEAR(dnlz_weights_vec[0],  0.036402571652033,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolerance(2.291652389327975, rel_tolorance);
	EXPECT_NEAR(dnlz_weights_vec[1],  2.291652389327975,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolerance(-0.000362096140397, rel_tolorance);
	EXPECT_NEAR(dnlz_weights_vec[2],  -0.000362096140397,  abs_tolorance);

	//Note that in the latest GPML3.5, derivatives wrt xu (covSEfact) does not support yet
	//In Shohun's implementation, the derivatives wrt xu are supported.
	//TParameter* lat_param=inf->m_gradient_parameters->get_parameter("inducing_features");
	//SGVector<float64_t> tmp=gradient->get_element(lat_param);
	//SGMatrix<float64_t> deriv_lat(tmp.vector, dim, m, false);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);

	SG_UNREF(inf);
}

#endif /* HAVE_LINALG_LIB */
