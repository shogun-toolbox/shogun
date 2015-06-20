/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
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
 *
 */
#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/labels/RegressionLabels.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/machine/gp/KLDualInferenceMethod.h>
#include <shogun/machine/gp/LogitDVGLikelihood.h>
#include <gtest/gtest.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

TEST(KLDualInferenceMethod,get_cholesky_logit_likelihood)
{
	// create some easy classification data:
	// y=sign(sqrt(x1.^2+x2.^2)-1)
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=0.8822936;
	feat_train(0, 1)=-0.7160792;
	feat_train(0, 2)=0.9178174;
	feat_train(0, 3)=-0.0135544;
	feat_train(0, 4)=-0.5275911;

	feat_train(1, 0)=-0.9597321;
	feat_train(1, 1)=0.0231289;
	feat_train(1, 2)=0.8284935;
	feat_train(1, 3)=0.0023812;
	feat_train(1, 4)=-0.7218931;

	lab_train[0]=1.0;
	lab_train[1]=-1.0;
	lab_train[2]=1.0;
	lab_train[3]=-1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	float64_t ell=2.0;
	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2*ell*ell);
	CZeroMean* mean=new CZeroMean();

	// logit likelihood
	CLogitDVGLikelihood* likelihood=new CLogitDVGLikelihood();

	// specify GP classification with KL inference
	CKLDualInferenceMethod* inf=new CKLDualInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	float64_t scale=2.0;
	inf->set_scale(scale);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/f9c9439698c1b0a0934a
	//
	// comparison of posterior cholesky with result:
	// L =
	//1.831315281491736   0.564406545992696   0.887800358037372   0.797470176790771   0.703766653832442
	//                0   1.332653329928429   0.443639773741460   0.534217181891372   0.492854017544880
	//                0                   0   1.585494546981023   0.374440968505748   0.084742698381037
	//                0                   0                   0   1.155117774908341   0.263962492491480
	//                0                   0                   0                   0   1.166811061103276


	SGMatrix<float64_t> L=inf->get_cholesky();
	float64_t rel_tolerance = 1e-3;
	float64_t abs_tolerance;

	abs_tolerance = CMath::get_abs_tolerance(1.831315281491736, rel_tolerance);
	EXPECT_NEAR(L(0,0),  1.831315281491736,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.564406545992696, rel_tolerance);
	EXPECT_NEAR(L(0,1),  0.564406545992696,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.887800358037372, rel_tolerance);
	EXPECT_NEAR(L(0,2),  0.887800358037372,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.797470176790771, rel_tolerance);
	EXPECT_NEAR(L(0,3),  0.797470176790771,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.703766653832442, rel_tolerance);
	EXPECT_NEAR(L(0,4),  0.703766653832442,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(L(1,0),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(1.332653329928429, rel_tolerance);
	EXPECT_NEAR(L(1,1),  1.332653329928429,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.443639773741460, rel_tolerance);
	EXPECT_NEAR(L(1,2),  0.443639773741460,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.534217181891372, rel_tolerance);
	EXPECT_NEAR(L(1,3),  0.534217181891372,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.492854017544880, rel_tolerance);
	EXPECT_NEAR(L(1,4),  0.492854017544880,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(L(2,0),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(L(2,1),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(1.585494546981023, rel_tolerance);
	EXPECT_NEAR(L(2,2),  1.585494546981023,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.374440968505748, rel_tolerance);
	EXPECT_NEAR(L(2,3),  0.374440968505748,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.084742698381037, rel_tolerance);
	EXPECT_NEAR(L(2,4),  0.084742698381037,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(L(3,0),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(L(3,1),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(L(3,2),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(1.155117774908341, rel_tolerance);
	EXPECT_NEAR(L(3,3),  1.155117774908341,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.263962492491480, rel_tolerance);
	EXPECT_NEAR(L(3,4),  0.263962492491480,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(L(4,0),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(L(4,1),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(L(4,2),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0, rel_tolerance);
	EXPECT_NEAR(L(4,3),  0,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(1.166811061103276, rel_tolerance);
	EXPECT_NEAR(L(4,4),  1.166811061103276,  abs_tolerance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLDualInferenceMethod,get_posterior_mean_logit_likelihood)
{
	// create some easy classification data:
	// y=sign(sqrt(x1.^2+x2.^2)-1)
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=0.8822936;
	feat_train(0, 1)=-0.7160792;
	feat_train(0, 2)=0.9178174;
	feat_train(0, 3)=-0.0135544;
	feat_train(0, 4)=-0.5275911;

	feat_train(1, 0)=-0.9597321;
	feat_train(1, 1)=0.0231289;
	feat_train(1, 2)=0.8284935;
	feat_train(1, 3)=0.0023812;
	feat_train(1, 4)=-0.7218931;

	lab_train[0]=1.0;
	lab_train[1]=-1.0;
	lab_train[2]=1.0;
	lab_train[3]=-1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// logit likelihood
	CLogitDVGLikelihood* likelihood=new CLogitDVGLikelihood();

	// specify GP classification with KL inference
	CKLDualInferenceMethod* inf=new CKLDualInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/f9c9439698c1b0a0934a
	//
	// comparison of posterior posterior_mean with result:
	// posterior_mean =
	//0.103590979631090
	//-0.833345382853119
	//0.166834580661953
	//-0.611796692525262
	//-0.751590461358533
	//

	SGVector<float64_t> posterior_mean=inf->get_posterior_mean();
	float64_t rel_tolerance = 1e-3;
	float64_t abs_tolerance;

	abs_tolerance = CMath::get_abs_tolerance(0.103590979631090, rel_tolerance);
	EXPECT_NEAR(posterior_mean[0],  0.103590979631090,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.833345382853119, rel_tolerance);
	EXPECT_NEAR(posterior_mean[1],  -0.833345382853119,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.166834580661953, rel_tolerance);
	EXPECT_NEAR(posterior_mean[2],  0.166834580661953,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.611796692525262, rel_tolerance);
	EXPECT_NEAR(posterior_mean[3],  -0.611796692525262,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.751590461358533, rel_tolerance);
	EXPECT_NEAR(posterior_mean[4],  -0.751590461358533,  abs_tolerance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLDualInferenceMethod,get_posterior_covariance_logit_likelihood)
{
	// create some easy classification data:
	// y=sign(sqrt(x1.^2+x2.^2)-1)
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=0.8822936;
	feat_train(0, 1)=-0.7160792;
	feat_train(0, 2)=0.9178174;
	feat_train(0, 3)=-0.0135544;
	feat_train(0, 4)=-0.5275911;

	feat_train(1, 0)=-0.9597321;
	feat_train(1, 1)=0.0231289;
	feat_train(1, 2)=0.8284935;
	feat_train(1, 3)=0.0023812;
	feat_train(1, 4)=-0.7218931;

	lab_train[0]=1.0;
	lab_train[1]=-1.0;
	lab_train[2]=1.0;
	lab_train[3]=-1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// logit likelihood
	CLogitDVGLikelihood* likelihood=new CLogitDVGLikelihood();

	// specify GP classification with KL inference
	CKLDualInferenceMethod* inf=new CKLDualInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/f9c9439698c1b0a0934a
	//
	// comparison of posterior posterior_covariance with result:
	// posterior_covariance =
	//0.592308700944687   0.012205069743884   0.058787302341417   0.154084602498963   0.131457795981947
	//0.012205069743884   0.568552559758170   0.040850804014863   0.363298426339381   0.348467345102775
	//0.058787302341417   0.040850804014863   0.591312958820500   0.190279380766388  -0.006726650771869
	//0.154084602498963   0.363298426339381   0.190279380766388   0.517484336632909   0.278169558166845
	//0.131457795981947   0.348467345102775  -0.006726650771869   0.278169558166845   0.576951507929158
	//

	SGMatrix<float64_t> posterior_covariance=inf->get_posterior_covariance();
	float64_t rel_tolerance = 1e-3;
	float64_t abs_tolerance;

	abs_tolerance = CMath::get_abs_tolerance(0.592308700944687, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(0,0),  0.592308700944687,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.012205069743884, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(0,1),  0.012205069743884,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.058787302341417, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(0,2),  0.058787302341417,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.154084602498963, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(0,3),  0.154084602498963,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.131457795981947, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(0,4),  0.131457795981947,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.012205069743884, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(1,0),  0.012205069743884,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.568552559758170, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(1,1),  0.568552559758170,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.040850804014863, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(1,2),  0.040850804014863,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.363298426339381, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(1,3),  0.363298426339381,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.348467345102775, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(1,4),  0.348467345102775,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.058787302341417, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(2,0),  0.058787302341417,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.040850804014863, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(2,1),  0.040850804014863,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.591312958820500, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(2,2),  0.591312958820500,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.190279380766388, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(2,3),  0.190279380766388,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.006726650771869, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(2,4),  -0.006726650771869,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.154084602498963, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(3,0),  0.154084602498963,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.363298426339381, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(3,1),  0.363298426339381,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.190279380766388, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(3,2),  0.190279380766388,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.517484336632909, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(3,3),  0.517484336632909,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.278169558166845, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(3,4),  0.278169558166845,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.131457795981947, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(4,0),  0.131457795981947,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.348467345102775, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(4,1),  0.348467345102775,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.006726650771869, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(4,2),  -0.006726650771869,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.278169558166845, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(4,3),  0.278169558166845,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.576951507929158, rel_tolerance);
	EXPECT_NEAR(posterior_covariance(4,4),  0.576951507929158,  abs_tolerance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLDualInferenceMethod,get_negative_marginal_likelihood_logit_likelihood)
{
	// create some easy classification data:
	// y=sign(sqrt(x1.^2+x2.^2)-1)
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=0.8822936;
	feat_train(0, 1)=-0.7160792;
	feat_train(0, 2)=0.9178174;
	feat_train(0, 3)=-0.0135544;
	feat_train(0, 4)=-0.5275911;

	feat_train(1, 0)=-0.9597321;
	feat_train(1, 1)=0.0231289;
	feat_train(1, 2)=0.8284935;
	feat_train(1, 3)=0.0023812;
	feat_train(1, 4)=-0.7218931;

	lab_train[0]=1.0;
	lab_train[1]=-1.0;
	lab_train[2]=1.0;
	lab_train[3]=-1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// logit likelihood
	CLogitDVGLikelihood* likelihood=new CLogitDVGLikelihood();

	// specify GP classification with KL inference
	CKLDualInferenceMethod* inf=new CKLDualInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/f9c9439698c1b0a0934a
	//
	// comparison of posterior negative marginal likelihood with
	// nlZ =
	//3.425144111752701
	float64_t nml=inf->get_negative_log_marginal_likelihood();
	float64_t rel_tolerance = 1e-3;
	float64_t abs_tolerance;

	abs_tolerance = CMath::get_abs_tolerance(3.425144111752701, rel_tolerance);
	EXPECT_NEAR(nml, 3.425144111752701, abs_tolerance);

	// clean up
	SG_UNREF(inf);
}

TEST(KLDualInferenceMethod,get_marginal_likelihood_derivatives_logit_likelihood)
{
	// create some easy classification data:
	// y=sign(sqrt(x1.^2+x2.^2)-1)
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=0.8822936;
	feat_train(0, 1)=-0.7160792;
	feat_train(0, 2)=0.9178174;
	feat_train(0, 3)=-0.0135544;
	feat_train(0, 4)=-0.5275911;

	feat_train(1, 0)=-0.9597321;
	feat_train(1, 1)=0.0231289;
	feat_train(1, 2)=0.8284935;
	feat_train(1, 3)=0.0023812;
	feat_train(1, 4)=-0.7218931;

	lab_train[0]=1.0;
	lab_train[1]=-1.0;
	lab_train[2]=1.0;
	lab_train[3]=-1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	float64_t ell=2.0;
	CGaussianKernel* kernel=new CGaussianKernel(10, 2*ell*ell);
	CZeroMean* mean=new CZeroMean();

	// logit likelihood
	CLogitDVGLikelihood* likelihood=new CLogitDVGLikelihood();

	// specify GP classification with KL inference
	CKLDualInferenceMethod* inf=new CKLDualInferenceMethod(kernel,
			features_train,	mean, labels_train, likelihood);

	float64_t scale=2.0;
	inf->set_scale(scale);

	// build parameter dictionary
	CMap<TParameter*, CSGObject*>* parameter_dictionary=new CMap<TParameter*, CSGObject*>();
	inf->build_gradient_parameter_dictionary(parameter_dictionary);

	// compute derivatives wrt parameters
	CMap<TParameter*, SGVector<float64_t> >* gradient=
		inf->get_negative_log_marginal_likelihood_derivatives(parameter_dictionary);

	// get parameters to compute derivatives
	TParameter* width_param=kernel->m_gradient_parameters->get_parameter("width");
	TParameter* scale_param=inf->m_gradient_parameters->get_parameter("scale");

	float64_t dnlZ_ell=4*ell*ell*(gradient->get_element(width_param))[0];
	float64_t dnlZ_sf2=scale*(gradient->get_element(scale_param))[0];

	//Reference result is generated from the Matlab code, which can be found at
	//https://gist.github.com/yorkerlin/f9c9439698c1b0a0934a
	//
	// comparison of partial derivatives of negative marginal likelihood with
	// cov =
	// 0.588696171221606
	// 0.031118659018803

	float64_t rel_tolerance = 1e-3;
	float64_t abs_tolerance;

	abs_tolerance = CMath::get_abs_tolerance(0.588696171221606, rel_tolerance);
	EXPECT_NEAR(dnlZ_ell, 0.588696171221606, abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.031118659018803, rel_tolerance);
	EXPECT_NEAR(dnlZ_sf2, 0.031118659018803, abs_tolerance);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);
	SG_UNREF(inf);
}

#endif /* HAVE_EIGEN3 */
