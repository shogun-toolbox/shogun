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
 * This code specifically adapted from function in varsgpLikelihood.m and varsgpPredict.m
 *
 * The reference paper is
 * Titsias, Michalis K.
 * "Variational learning of inducing variables in sparse Gaussian processes."
 * International Conference on Artificial Intelligence and Statistics. 2009.
 *
 */
#include <shogun/machine/gp/SparseVGInferenceMethod.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CSparseVGInferenceMethod::CSparseVGInferenceMethod() : CSingleSparseInferenceBase()
{
	init();
}

CSparseVGInferenceMethod::CSparseVGInferenceMethod(CKernel* kern, CFeatures* feat,
		CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod, CFeatures* lat)
		: CSingleSparseInferenceBase(kern, feat, m, lab, mod, lat)
{
	init();
}

void CSparseVGInferenceMethod::init()
{
}

CSparseVGInferenceMethod::~CSparseVGInferenceMethod()
{
}
void CSparseVGInferenceMethod::compute_gradient()
{
	CInferenceMethod::compute_gradient();

	if (!m_gradient_update)
	{
		update_deriv();
		m_gradient_update=true;
		update_parameter_hash();
	}
}

void CSparseVGInferenceMethod::update()
{
	SG_DEBUG("entering\n");

	CInferenceMethod::update();
	update_chol();
	update_alpha();
	m_gradient_update=false;
	update_parameter_hash();

	SG_DEBUG("leaving\n");
}

CSparseVGInferenceMethod* CSparseVGInferenceMethod::obtain_from_generic(
		CInferenceMethod* inference)
{
	if (inference==NULL)
		return NULL;

	if (inference->get_inference_type()!=INF_KL_SPARSE_REGRESSION)
		SG_SERROR("Provided inference is not of type CSparseVGInferenceMethod!\n")

	SG_REF(inference);
	return (CSparseVGInferenceMethod*)inference;
}

void CSparseVGInferenceMethod::check_members() const
{
	CSingleSparseInferenceBase::check_members();

	REQUIRE(m_model->get_model_type()==LT_GAUSSIAN,
			"SparseVG inference method can only use Gaussian likelihood function\n")
	REQUIRE(m_labels->get_label_type()==LT_REGRESSION, "Labels must be type "
			"of CRegressionLabels\n")
}

SGVector<float64_t> CSparseVGInferenceMethod::get_diagonal_vector()
{
	return SGVector<float64_t>();
}

float64_t CSparseVGInferenceMethod::get_negative_log_marginal_likelihood()
{
	if (parameter_hash_changed())
		update();

	Map<MatrixXd> eigen_inv_La(m_inv_La.matrix, m_inv_La.num_rows, m_inv_La.num_cols);
	Map<VectorXd> eigen_ktrtr_diag(m_ktrtr_diag.vector, m_ktrtr_diag.vlen);

	//F012 =-(model.n-model.m)*model.Likelihood.logtheta-0.5*model.n*log(2*pi)-(0.5/sigma2)*(model.yy)-sum(log(diag(La)));
	float64_t neg_f012=(m_ktru.num_cols-m_ktru.num_rows)*CMath::log(m_sigma2)/2.0
		+0.5*m_ktru.num_cols*CMath::log(2*CMath::PI)+
		0.5*m_yy/(m_sigma2)-eigen_inv_La.diagonal().array().log().sum();

	//F3 = (0.5/sigma2)*(yKnmInvLmInvLa*yKnmInvLmInvLa');
	float64_t neg_f3=-m_f3;

	//model.TrKnn = sum(model.diagKnn);
	//TrK = - (0.5/sigma2)*(model.TrKnn  - sum(diag(C)) );
	float64_t neg_trk=-m_trk;

	//F = F012 + F3 + TrK;
	//F = - F;
	return neg_f012+neg_f3+neg_trk;
}

void CSparseVGInferenceMethod::update_chol()
{
	// get the sigma variable from the Gaussian likelihood model
	CGaussianLikelihood* lik=CGaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);
	m_sigma2=sigma*sigma;

	//m-by-m matrix
	Map<MatrixXd> eigen_kuu(m_kuu.matrix, m_kuu.num_rows, m_kuu.num_cols);

	//m-by-n matrix
	Map<MatrixXd> eigen_ktru(m_ktru.matrix, m_ktru.num_rows, m_ktru.num_cols);

	Map<VectorXd> eigen_ktrtr_diag(m_ktrtr_diag.vector, m_ktrtr_diag.vlen);

	//Lm = chol(model.Kmm + model.jitter*eye(model.m))
	LLT<MatrixXd> Luu(eigen_kuu*CMath::exp(m_log_scale*2.0)+CMath::exp(m_log_ind_noise)*MatrixXd::Identity(
		m_kuu.num_rows, m_kuu.num_cols));
	m_inv_Lm=SGMatrix<float64_t>(Luu.rows(), Luu.cols());
	Map<MatrixXd> eigen_inv_Lm(m_inv_Lm.matrix, m_inv_Lm.num_rows, m_inv_Lm.num_cols);
	//invLm = Lm\eye(model.m); 
	eigen_inv_Lm=Luu.matrixU().solve(MatrixXd::Identity(m_kuu.num_rows, m_kuu.num_cols));

	m_Knm_inv_Lm=SGMatrix<float64_t>(m_ktru.num_cols, m_ktru.num_rows);
	Map<MatrixXd> eigen_Knm_inv_Lm(m_Knm_inv_Lm.matrix, m_Knm_inv_Lm.num_rows, m_Knm_inv_Lm.num_cols);
	//KnmInvLm = model.Knm*invLm;      
	eigen_Knm_inv_Lm=(eigen_ktru.transpose()*CMath::exp(m_log_scale*2.0))*eigen_inv_Lm;

	m_Tmm=SGMatrix<float64_t>(m_kuu.num_rows, m_kuu.num_cols);
	Map<MatrixXd> eigen_C(m_Tmm.matrix, m_Tmm.num_rows, m_Tmm.num_cols);
	//C = KnmInvLm'*KnmInvLm; 
	eigen_C=eigen_Knm_inv_Lm.transpose()*eigen_Knm_inv_Lm;

	m_inv_La=SGMatrix<float64_t>(m_kuu.num_rows, m_kuu.num_cols);
	Map<MatrixXd> eigen_inv_La(m_inv_La.matrix, m_inv_La.num_rows, m_inv_La.num_cols);
	//A = sigma2*eye(model.m) + C;    
	LLT<MatrixXd> chol_A(m_sigma2*MatrixXd::Identity(m_kuu.num_rows, m_kuu.num_cols)+eigen_C);

	//La = chol(A);
	//invLa =  La\eye(model.m);  
	eigen_inv_La=chol_A.matrixU().solve(MatrixXd::Identity(m_kuu.num_rows, m_kuu.num_cols));

	//L=-invLm*invLm' +  sigma2*(invLm*invLa*invLa'*invLm');
	m_L=SGMatrix<float64_t>(m_kuu.num_rows, m_kuu.num_cols);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);
	eigen_L=eigen_inv_Lm*(
		m_sigma2*eigen_inv_La*eigen_inv_La.transpose()-MatrixXd::Identity(m_kuu.num_rows, m_kuu.num_cols)
		)*eigen_inv_Lm.transpose();

	//TrK = - (0.5/sigma2)*(model.TrKnn  - sum(diag(C)) );
	m_trk=-0.5/(m_sigma2)*(eigen_ktrtr_diag.array().sum()*CMath::exp(m_log_scale*2.0)
		-eigen_C.diagonal().array().sum());
}

void CSparseVGInferenceMethod::update_alpha()
{
	Map<MatrixXd> eigen_Knm_inv_Lm(m_Knm_inv_Lm.matrix, m_Knm_inv_Lm.num_rows, m_Knm_inv_Lm.num_cols);
	Map<MatrixXd> eigen_inv_La(m_inv_La.matrix, m_inv_La.num_rows, m_inv_La.num_cols);
	Map<MatrixXd> eigen_inv_Lm(m_inv_Lm.matrix, m_inv_Lm.num_rows, m_inv_Lm.num_cols);

	SGVector<float64_t> y=((CRegressionLabels*) m_labels)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);
	SGVector<float64_t> m=m_mean->get_mean_vector(m_features);
	Map<VectorXd> eigen_m(m.vector, m.vlen);

	//yKnmInvLm = (model.y'*KnmInvLm);  
	//yKnmInvLmInvLa = yKnmInvLm*invLa;     
	VectorXd y_cor=eigen_y-eigen_m;
	VectorXd eigen_y_Knm_inv_Lm_inv_La_transpose=eigen_inv_La.transpose()*(
		eigen_Knm_inv_Lm.transpose()*y_cor);
	//alpha = invLm*invLa*yKnmInvLmInvLa'; 
	m_alpha=SGVector<float64_t>(m_kuu.num_rows);
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);
	eigen_alpha=eigen_inv_Lm*eigen_inv_La*eigen_y_Knm_inv_Lm_inv_La_transpose;

	m_yy=y_cor.dot(y_cor);
	//F3 = (0.5/sigma2)*(yKnmInvLmInvLa*yKnmInvLmInvLa');
	m_f3=0.5*eigen_y_Knm_inv_Lm_inv_La_transpose.dot(eigen_y_Knm_inv_Lm_inv_La_transpose)/m_sigma2;
}

void CSparseVGInferenceMethod::update_deriv()
{
	Map<MatrixXd> eigen_inv_La(m_inv_La.matrix, m_inv_La.num_rows, m_inv_La.num_cols);
	Map<MatrixXd> eigen_inv_Lm(m_inv_Lm.matrix, m_inv_Lm.num_rows, m_inv_Lm.num_cols);
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);
	//m-by-n matrix
	Map<MatrixXd> eigen_ktru(m_ktru.matrix, m_ktru.num_rows, m_ktru.num_cols);
	Map<MatrixXd> eigen_C(m_Tmm.matrix, m_Tmm.num_rows, m_Tmm.num_cols);

	//m_Tmm=SGMatrix<float64_t>(m_kuu.num_rows, m_kuu.num_cols);
	m_Tnm=SGMatrix<float64_t>(m_ktru.num_cols, m_ktru.num_rows);

	Map<MatrixXd> eigen_Tmm(m_Tmm.matrix, m_Tmm.num_rows, m_Tmm.num_cols);
	Map<MatrixXd> eigen_Tnm(m_Tnm.matrix, m_Tnm.num_rows, m_Tnm.num_cols);

	CGaussianLikelihood* lik=CGaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);
	m_sigma2=sigma*sigma;

	//invLmInvLa = invLm*invLa;  
	//invA = invLmInvLa*invLmInvLa';    
	//yKnmInvA = yKnmInvLmInvLa*invLmInvLa'; 
	//invKmm = invLm*invLm'; 
	//Tmm = sigma2*invA + yKnmInvA'*yKnmInvA;  
	//Tmm = invKmm - Tmm;
	MatrixXd Tmm=-eigen_L-eigen_alpha*eigen_alpha.transpose();
	
	//Tnm = model.Knm*Tmm;  
	eigen_Tnm=(eigen_ktru.transpose()*CMath::exp(m_log_scale*2.0))*Tmm;

	//Tmm = Tmm - (invLm*(C*invLm'))/sigma2; 
	eigen_Tmm = Tmm - (eigen_inv_Lm*eigen_C*eigen_inv_Lm.transpose()/m_sigma2);

	SGVector<float64_t> y=((CRegressionLabels*) m_labels)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);
	SGVector<float64_t> m=m_mean->get_mean_vector(m_features);
	Map<VectorXd> eigen_m(m.vector, m.vlen);

	//Tnm = Tnm + (model.y*yKnmInvA); 
	eigen_Tnm += (eigen_y-eigen_m)*eigen_alpha.transpose();
}

SGVector<float64_t> CSparseVGInferenceMethod::get_posterior_mean()
{
	return SGVector<float64_t>();
}

SGMatrix<float64_t> CSparseVGInferenceMethod::get_posterior_covariance()
{
	return SGMatrix<float64_t>();
}

SGVector<float64_t> CSparseVGInferenceMethod::get_derivative_wrt_likelihood_model(
		const TParameter* param)
{
	REQUIRE(!strcmp(param->m_name, "log_sigma"), "Can't compute derivative of "
			"the nagative log marginal likelihood wrt %s.%s parameter\n",
			m_model->get_name(), param->m_name)

	SGVector<float64_t> dlik(1);

	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);
	Map<MatrixXd> eigen_inv_La(m_inv_La.matrix, m_inv_La.num_rows, m_inv_La.num_cols);
	Map<MatrixXd> eigen_ktru(m_ktru.matrix, m_ktru.num_rows, m_ktru.num_cols);
	Map<MatrixXd> eigen_kuu(m_kuu.matrix, m_kuu.num_rows, m_kuu.num_cols);
	//yKnmInvLmInvLainvLa = yKnmInvLmInvLa*invLa';
	//sigma2aux = sigma2*sum(sum(invLa.*invLa))  + yKnmInvLmInvLainvLa*yKnmInvLmInvLainvLa';
	float64_t sigma2aux= m_sigma2*eigen_inv_La.cwiseProduct(eigen_inv_La).array().sum()+
		eigen_alpha.transpose()*(eigen_kuu*CMath::exp(m_log_scale*2.0)
			+CMath::exp(m_log_ind_noise)*MatrixXd::Identity(
		m_kuu.num_rows, m_kuu.num_cols))*eigen_alpha;
	//Dlik_neg = - (model.n-model.m) + model.yy/sigma2 - 2*F3 - sigma2aux - 2*TrK;
	
	dlik[0]=(m_ktru.num_cols-m_ktru.num_rows)-m_yy/m_sigma2+2.0*m_f3+sigma2aux+2.0*m_trk;
	return dlik;
}

SGVector<float64_t> CSparseVGInferenceMethod::get_derivative_wrt_inducing_features(
	const TParameter* param)
{
	//[DXu DXunm] = kernelSparseGradInd(model, Tmm, Tnm);
    //DXu_neg = DXu + DXunm/model.sigma2;
	
	Map<MatrixXd> eigen_Tmm(m_Tmm.matrix, m_Tmm.num_rows, m_Tmm.num_cols);
	Map<MatrixXd> eigen_Tnm(m_Tnm.matrix, m_Tnm.num_rows, m_Tnm.num_cols);

	int32_t dim=m_inducing_features.num_rows;
	int32_t num_samples=m_inducing_features.num_cols;
	SGVector<float64_t>deriv_lat(dim*num_samples);
	deriv_lat.zero();

	m_lock->lock();
	CFeatures *inducing_features=get_inducing_features();
	//asymtric part (related to xu and x)
	m_kernel->init(inducing_features, m_features);
	for(int32_t lat_idx=0; lat_idx<eigen_Tnm.cols(); lat_idx++)
	{
		Map<VectorXd> deriv_lat_col_vec(deriv_lat.vector+lat_idx*dim,dim);
		//p by n
		SGMatrix<float64_t> deriv_mat=m_kernel->get_parameter_gradient(param, lat_idx);
		Map<MatrixXd> eigen_deriv_mat(deriv_mat.matrix, deriv_mat.num_rows, deriv_mat.num_cols);
		//DXunm/model.sigma2;
		deriv_lat_col_vec+=eigen_deriv_mat*(-CMath::exp(m_log_scale*2.0)/m_sigma2*eigen_Tnm.col(lat_idx));
	}
	m_lock->unlock();

	m_lock->lock();
	//symtric part (related to xu and xu)
	m_kernel->init(inducing_features, inducing_features);
	for(int32_t lat_lidx=0; lat_lidx<eigen_Tmm.cols(); lat_lidx++)
	{
		Map<VectorXd> deriv_lat_col_vec(deriv_lat.vector+lat_lidx*dim,dim);
		//p by n
		SGMatrix<float64_t> deriv_mat=m_kernel->get_parameter_gradient(param, lat_lidx);
		Map<MatrixXd> eigen_deriv_mat(deriv_mat.matrix, deriv_mat.num_rows, deriv_mat.num_cols);
		//DXu
		deriv_lat_col_vec+=eigen_deriv_mat*(-CMath::exp(m_log_scale*2.0)*eigen_Tmm.col(lat_lidx));
	}
	SG_UNREF(inducing_features);
	m_lock->unlock();
	return deriv_lat;
}

SGVector<float64_t> CSparseVGInferenceMethod::get_derivative_wrt_inducing_noise(
	const TParameter* param)
{
	REQUIRE(param, "Param not set\n");
	REQUIRE(!strcmp(param->m_name, "log_inducing_noise"), "Can't compute derivative of "
			"the nagative log marginal likelihood wrt %s.%s parameter\n",
			get_name(), param->m_name)

	Map<MatrixXd> eigen_Tmm(m_Tmm.matrix, m_Tmm.num_rows, m_Tmm.num_cols);
	SGVector<float64_t> result(1);
	result[0]=-0.5*CMath::exp(m_log_ind_noise)*eigen_Tmm.diagonal().array().sum();
	return result;
}

float64_t CSparseVGInferenceMethod::get_derivative_related_cov(SGVector<float64_t> ddiagKi,
	SGMatrix<float64_t> dKuui, SGMatrix<float64_t> dKui)
{
	Map<VectorXd> eigen_ddiagKi(ddiagKi.vector, ddiagKi.vlen);
	Map<MatrixXd> eigen_dKuui(dKuui.matrix, dKuui.num_rows, dKuui.num_cols);
	Map<MatrixXd> eigen_dKui(dKui.matrix, dKui.num_rows, dKui.num_cols);

	Map<MatrixXd> eigen_Tmm(m_Tmm.matrix, m_Tmm.num_rows, m_Tmm.num_cols);
	Map<MatrixXd> eigen_Tnm(m_Tnm.matrix, m_Tnm.num_rows, m_Tnm.num_cols);

	//[Dkern Dkernnm DTrKnn] = kernelSparseGradHyp(model, Tmm, Tnm); 
	//Dkern_neg = 0.5*Dkern + Dkernnm/model.sigma2 - (0.5/model.sigma2)*DTrKnn;
	float64_t dkern= -0.5*eigen_dKuui.cwiseProduct(eigen_Tmm).sum()
		-eigen_dKui.cwiseProduct(eigen_Tnm.transpose()).sum()/m_sigma2
		+0.5*eigen_ddiagKi.array().sum()/m_sigma2;
	return dkern;
}

SGVector<float64_t> CSparseVGInferenceMethod::get_derivative_wrt_mean(
	const TParameter* param)
{
	REQUIRE(param, "Param not set\n");
	SGVector<float64_t> result;
	int64_t len=const_cast<TParameter *>(param)->m_datatype.get_num_elements();
	result=SGVector<float64_t>(len);

	SGVector<float64_t> y=((CRegressionLabels*) m_labels)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);
	SGVector<float64_t> m=m_mean->get_mean_vector(m_features);
	Map<VectorXd> eigen_m(m.vector, m.vlen);
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);

	//m-by-n matrix
	Map<MatrixXd> eigen_ktru(m_ktru.matrix, m_ktru.num_rows, m_ktru.num_cols);

	for (index_t i=0; i<result.vlen; i++)
	{
		SGVector<float64_t> dmu=m_mean->get_parameter_derivative(m_features, param, i);
		Map<VectorXd> eigen_dmu(dmu.vector, dmu.vlen);

		result[i]=eigen_dmu.dot(eigen_ktru.transpose()*CMath::exp(m_log_scale*2.0)
			*eigen_alpha+(eigen_m-eigen_y))/m_sigma2;
	}
	return result;
}
#endif /* HAVE_EIGEN3 */
