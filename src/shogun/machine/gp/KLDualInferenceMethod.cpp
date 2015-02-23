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
 * Code adapted from 
 * http://hannes.nickisch.org/code/approxXX.tar.gz
 * and Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 * and the reference paper is
 * Mohammad Emtiyaz Khan, Aleksandr Y. Aravkin, Michael P. Friedlander, Matthias Seeger
 * Fast Dual Variational Inference for Non-Conjugate Latent Gaussian Models. ICML2013*
 *
 * This code specifically adapted from function in approxKL.m and infKL.m
 */

#include <shogun/machine/gp/KLDualInferenceMethod.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Math.h>
#include <shogun/machine/gp/MatrixOperations.h>
#include <shogun/machine/gp/DualVariationalGaussianLikelihood.h>
#include <shogun/labels/BinaryLabels.h>

using namespace Eigen;

namespace shogun
{

CKLDualInferenceMethod::CKLDualInferenceMethod() : CKLInferenceMethod()
{
	init();
}

CKLDualInferenceMethod::CKLDualInferenceMethod(CKernel* kern,
		CFeatures* feat, CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod)
		: CKLInferenceMethod(kern, feat, m, lab, mod)
{
	init();
}

CKLDualInferenceMethod* CKLDualInferenceMethod::obtain_from_generic(
		CInferenceMethod* inference)
{
	REQUIRE(inference, "Inference pointer not set.\n");

	if (inference->get_inference_type()!=INF_KL_DUAL)
		SG_SERROR("Provided inference is not of type CKLDualInferenceMethod!\n")

	SG_REF(inference);
	return (CKLDualInferenceMethod*)inference;
}

SGVector<float64_t> CKLDualInferenceMethod::get_alpha()
{
	if (parameter_hash_changed())
		update();

	SGVector<float64_t> result(m_alpha);
	return result;
}

CKLDualInferenceMethod::~CKLDualInferenceMethod()
{
}

void CKLDualInferenceMethod::check_dual_inference(CLikelihoodModel* mod) const
{
	CDualVariationalGaussianLikelihood * lik=dynamic_cast<CDualVariationalGaussianLikelihood *>(mod);
	REQUIRE(lik,
		"The provided likelihood model is not a variational dual Likelihood model.\n");
}

void CKLDualInferenceMethod::set_model(CLikelihoodModel* mod)
{
	check_dual_inference(mod);
	CKLInferenceMethod::set_model(mod);
}

CDualVariationalGaussianLikelihood* CKLDualInferenceMethod::get_dual_variational_likelihood() const
{
	check_dual_inference(m_model);
	CDualVariationalGaussianLikelihood * lik=dynamic_cast<CDualVariationalGaussianLikelihood *>(m_model);
	return lik;
}

void CKLDualInferenceMethod::init()
{
	SG_ADD(&m_W, "W",
		"noise matrix W",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_sW, "sW",
		"Square root of noise matrix W",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_dv, "dv",
		"the gradient of the variational expection wrt sigma2",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_df, "df",
		"the gradient of the variational expection wrt mu",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_is_dual_valid, "is_dual_valid",
		"whether the lambda (m_W) is valid or not",
		MS_NOT_AVAILABLE);
	m_is_dual_valid=false;
}

bool CKLDualInferenceMethod::lbfgs_precompute()
{
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	CDualVariationalGaussianLikelihood *lik= get_dual_variational_likelihood();
	Map<VectorXd> eigen_W(m_W.vector, m_W.vlen);

	lik->set_dual_parameters(m_W, m_labels);
	m_is_dual_valid=lik->dual_parameters_valid();

	if (!m_is_dual_valid)
		return false;

	//construct alpha
	m_alpha=lik->get_mu_dual_parameter();
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);
	eigen_alpha=-eigen_alpha;

	Map<VectorXd> eigen_sW(m_sW.vector, m_sW.vlen);
	eigen_sW=eigen_W.array().sqrt().matrix();

	m_L=CMatrixOperations::get_choleksy(m_W, m_sW, m_ktrtr, m_scale);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);
	
	//solve L'*V=diag(sW)*K
	Map<MatrixXd> eigen_V(m_V.matrix, m_V.num_rows, m_V.num_cols);
	eigen_V=eigen_L.triangularView<Upper>().adjoint().solve(eigen_sW.asDiagonal()*eigen_K*CMath::sq(m_scale));
	Map<VectorXd> eigen_s2(m_s2.vector, m_s2.vlen);
	//Sigma=inv(inv(K)+diag(W))=K-K*diag(sW)*inv(L)'*inv(L)*diag(sW)*K
	//v=abs(diag(Sigma))
	eigen_s2=(eigen_K.diagonal().array()*CMath::sq(m_scale)-(eigen_V.array().pow(2).colwise().sum().transpose())).abs().matrix();

	//construct mu
	SGVector<float64_t> mean=m_mean->get_mean_vector(m_features);
	Map<VectorXd> eigen_mean(mean.vector, mean.vlen);
	Map<VectorXd> eigen_mu(m_mu.vector, m_mu.vlen);
	//mu=K*alpha+m
	eigen_mu=eigen_K*CMath::sq(m_scale)*eigen_alpha+eigen_mean;
	return true;
}

float64_t CKLDualInferenceMethod::get_dual_objective_wrt_parameters()
{
	if (!m_is_dual_valid)
		return CMath::INFTY;

	SGVector<float64_t> mean=m_mean->get_mean_vector(m_features);
	Map<VectorXd> eigen_mean(mean.vector, mean.vlen);
	Map<VectorXd> eigen_mu(m_mu.vector, m_mu.vlen);
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);

	CDualVariationalGaussianLikelihood *lik= get_dual_variational_likelihood();

	float64_t a=SGVector<float64_t>::sum(lik->get_dual_objective_value());
	float64_t result=0.5*eigen_alpha.dot(eigen_mu-eigen_mean)+a;
	result+=eigen_mean.dot(eigen_alpha);
	result-=eigen_L.diagonal().array().log().sum();

	return result;
}

void CKLDualInferenceMethod::get_gradient_of_dual_objective_wrt_parameters(SGVector<float64_t> gradient)
{
	REQUIRE(gradient.vlen==m_alpha.vlen,
		"The length of gradients (%d) should the same as the length of parameters (%d)\n",
		gradient.vlen, m_alpha.vlen);

	if (!m_is_dual_valid)
		return;

	Map<VectorXd> eigen_gradient(gradient.vector, gradient.vlen);

	CDualVariationalGaussianLikelihood *lik= get_dual_variational_likelihood();

	TParameter* lambda_param=lik->m_parameters->get_parameter("lambda");
	SGVector<float64_t>d_lambda=lik->get_dual_first_derivative(lambda_param);
	Map<VectorXd> eigen_d_lambda(d_lambda.vector, d_lambda.vlen);

	Map<VectorXd> eigen_mu(m_mu.vector, m_mu.vlen);
	Map<VectorXd> eigen_s2(m_s2.vector, m_s2.vlen);

	eigen_gradient=-eigen_mu-0.5*eigen_s2+eigen_d_lambda;
}

float64_t CKLDualInferenceMethod::get_nlml_wrapper(SGVector<float64_t> alpha, SGVector<float64_t> mu, SGMatrix<float64_t> L)
{
	Map<MatrixXd> eigen_L(L.matrix, L.num_rows, L.num_cols);
	Map<VectorXd> eigen_alpha(alpha.vector, alpha.vlen);
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	//get mean vector and create eigen representation of it
	SGVector<float64_t> mean=m_mean->get_mean_vector(m_features);
	Map<VectorXd> eigen_mu(mu.vector, mu.vlen);
	Map<VectorXd> eigen_mean(mean.vector, mean.vlen);

	CDualVariationalGaussianLikelihood *lik=get_dual_variational_likelihood();

	SGVector<float64_t>lab=((CBinaryLabels*)m_labels)->get_labels();
	Map<VectorXd> eigen_lab(lab.vector, lab.vlen);

	float64_t a=SGVector<float64_t>::sum(lik->get_variational_expection());

	float64_t trace=0;
	//L_inv=L\eye(n);
	//trace(L_inv'*L_inv)   %V*inv(K)
	MatrixXd eigen_t=eigen_L.triangularView<Upper>().adjoint().solve(MatrixXd::Identity(eigen_L.rows(),eigen_L.cols()));

	for(index_t idx=0; idx<eigen_t.rows(); idx++)
		trace +=(eigen_t.col(idx).array().pow(2)).sum();

	//nlZ = -a -logdet(V*inv(K))/2 -n/2 +(alpha'*K*alpha)/2 +trace(V*inv(K))/2;	
	float64_t result=-a+eigen_L.diagonal().array().log().sum();

	result+=0.5*(-eigen_K.rows()+eigen_alpha.dot(eigen_mu-eigen_mean)+trace);
	return result;
}

float64_t CKLDualInferenceMethod::get_negative_log_marginal_likelihood_helper()
{
	CDualVariationalGaussianLikelihood *lik=get_dual_variational_likelihood();
	bool status = lik->set_variational_distribution(m_mu, m_s2, m_labels);
	if (status)
		return get_nlml_wrapper(m_alpha, m_mu, m_L);
	return CMath::NOT_A_NUMBER;
}

float64_t CKLDualInferenceMethod::get_derivative_related_cov(SGMatrix<float64_t> dK)
{
	Map<MatrixXd> eigen_dK(dK.matrix, dK.num_rows, dK.num_cols);
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<VectorXd> eigen_W(m_W.vector, m_W.vlen);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);
	Map<VectorXd> eigen_sW(m_sW.vector, m_sW.vlen);
	Map<MatrixXd> eigen_Sigma(m_Sigma.matrix, m_Sigma.num_rows, m_Sigma.num_cols);
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);

	Map<VectorXd> eigen_dv(m_dv.vector, m_dv.vlen);
	Map<VectorXd> eigen_df(m_df.vector, m_df.vlen);

	index_t len=m_W.vlen;
	//U=inv(L')*diag(sW)
	MatrixXd eigen_U=eigen_L.triangularView<Upper>().adjoint().solve(MatrixXd(eigen_sW.asDiagonal()));
	//A=I-K*diag(sW)*inv(L)*inv(L')*diag(sW)
	Map<MatrixXd> eigen_V(m_V.matrix, m_V.num_rows, m_V.num_cols);
	MatrixXd eigen_A=MatrixXd::Identity(len, len)-eigen_V.transpose()*eigen_U;

	//AdK = A*dK;
	MatrixXd AdK=eigen_A*eigen_dK;

	//z = diag(AdK) + sum(A.*AdK,2) - sum(A'.*AdK,1)';
	VectorXd z=AdK.diagonal()+(eigen_A.array()*AdK.array()).rowwise().sum().matrix()
		-(eigen_A.transpose().array()*AdK.array()).colwise().sum().transpose().matrix();

	float64_t result=eigen_alpha.dot(eigen_dK*(eigen_alpha/2.0-eigen_df))-z.dot(eigen_dv);

	return result;
}

void CKLDualInferenceMethod::update_alpha()
{
	float64_t nlml_new=0;
	float64_t nlml_def=0;

	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	CDualVariationalGaussianLikelihood *lik= get_dual_variational_likelihood();

	if (m_alpha.vlen == m_labels->get_num_labels())
	{
		nlml_new=get_negative_log_marginal_likelihood_helper();
		index_t len=m_labels->get_num_labels();
		SGVector<float64_t> W_tmp(len);
		Map<VectorXd> eigen_W(W_tmp.vector, W_tmp.vlen);
		eigen_W.fill(0.5);
		SGVector<float64_t> sW_tmp(len);
		Map<VectorXd> eigen_sW(sW_tmp.vector, sW_tmp.vlen);
		eigen_sW=eigen_W.array().sqrt().matrix();
		SGMatrix<float64_t> L_tmp=CMatrixOperations::get_choleksy(W_tmp, sW_tmp, m_ktrtr, m_scale);
		Map<MatrixXd> eigen_L(L_tmp.matrix, L_tmp.num_rows, L_tmp.num_cols);

		lik->set_dual_parameters(W_tmp, m_labels);

		//construct alpha
		SGVector<float64_t> alpha_tmp=lik->get_mu_dual_parameter();
		Map<VectorXd> eigen_alpha(alpha_tmp.vector, alpha_tmp.vlen);
		eigen_alpha=-eigen_alpha;
		//construct mu
		SGVector<float64_t> mean=m_mean->get_mean_vector(m_features);
		Map<VectorXd> eigen_mean(mean.vector, mean.vlen);
		SGVector<float64_t> mu_tmp(len);
		Map<VectorXd> eigen_mu(mu_tmp.vector, mu_tmp.vlen);
		//mu=K*alpha+m
		eigen_mu=eigen_K*CMath::sq(m_scale)*eigen_alpha+eigen_mean;
		//construct s2
		MatrixXd eigen_V=eigen_L.triangularView<Upper>().adjoint().solve(eigen_sW.asDiagonal()*eigen_K*CMath::sq(m_scale));
		SGVector<float64_t> s2_tmp(len);
		Map<VectorXd> eigen_s2(s2_tmp.vector, s2_tmp.vlen);
		eigen_s2=(eigen_K.diagonal().array()*CMath::sq(m_scale)-(eigen_V.array().pow(2).colwise().sum().transpose())).abs().matrix();

		lik->set_variational_distribution(mu_tmp, s2_tmp, m_labels);

		nlml_def=get_nlml_wrapper(alpha_tmp, mu_tmp, L_tmp);

		if (nlml_new<=nlml_def)
		{
			lik->set_dual_parameters(m_W, m_labels);
			lik->set_variational_distribution(m_mu, m_s2, m_labels);
		}
	}

	if (m_alpha.vlen != m_labels->get_num_labels() || nlml_def<nlml_new)
	{
		if(m_alpha.vlen != m_labels->get_num_labels())
			m_alpha = SGVector<float64_t>(m_labels->get_num_labels());

		index_t len=m_alpha.vlen;

		m_W=SGVector<float64_t>(len);
		for (index_t i=0; i<m_W.vlen; i++)
			m_W[i]=0.5;

		lik->set_dual_parameters(m_W, m_labels);
		m_sW=SGVector<float64_t>(len);
		m_mu=SGVector<float64_t>(len);
		m_s2=SGVector<float64_t>(len);
		m_Sigma=SGMatrix<float64_t>(len, len);
		m_V=SGMatrix<float64_t>(len, len);
	}

	nlml_new=lbfgs_optimization();
	lik->set_variational_distribution(m_mu, m_s2, m_labels);
	TParameter* s2_param=lik->m_parameters->get_parameter("sigma2");
	m_dv=lik->get_variational_first_derivative(s2_param);
	TParameter* mu_param=lik->m_parameters->get_parameter("mu");
	m_df=lik->get_variational_first_derivative(mu_param);
}

float64_t CKLDualInferenceMethod::adjust_step(void *obj,
	const float64_t *parameters, const float64_t *direction,
	const int dim, const float64_t step)
{
	/* Note that parameters = parameters_pre_iter - step * gradient_pre_iter */
	CKLDualInferenceMethod * obj_prt
		= static_cast<CKLDualInferenceMethod *>(obj);

	ASSERT(obj_prt != NULL);

	float64_t *non_const_direction=const_cast<float64_t *>(direction);
	SGVector<float64_t> sg_direction(non_const_direction, dim, false);

	CDualVariationalGaussianLikelihood* lik= obj_prt->get_dual_variational_likelihood();

	float64_t adjust_stp=lik->adjust_step_wrt_dual_parameter(sg_direction, step);
	return adjust_stp;
}

float64_t CKLDualInferenceMethod::evaluate(void *obj, const float64_t *parameters,
	float64_t *gradient, const int dim, const float64_t step)
{
	/* Note that parameters = parameters_pre_iter - step * gradient_pre_iter */
	CKLDualInferenceMethod * obj_prt
		= static_cast<CKLDualInferenceMethod *>(obj);

	ASSERT(obj_prt != NULL);

	bool status=obj_prt->lbfgs_precompute();
	if (status)
	{
		float64_t nlml=obj_prt->get_dual_objective_wrt_parameters();

		SGVector<float64_t> sg_gradient(gradient, dim, false);
		Map<VectorXd> eigen_g(sg_gradient.vector, sg_gradient.vlen);
		obj_prt->get_gradient_of_dual_objective_wrt_parameters(sg_gradient);

		return nlml;
	}
	return CMath::NOT_A_NUMBER;
}

float64_t CKLDualInferenceMethod::lbfgs_optimization()
{
	lbfgs_parameter_t lbfgs_param;
	lbfgs_param.m = m_m;
	lbfgs_param.max_linesearch = m_max_linesearch;
	lbfgs_param.linesearch = m_linesearch;
	lbfgs_param.max_iterations = m_max_iterations;
	lbfgs_param.delta = m_delta;
	lbfgs_param.past = m_past;
	lbfgs_param.epsilon = m_epsilon;
	lbfgs_param.min_step = m_min_step;
	lbfgs_param.max_step = m_max_step;
	lbfgs_param.ftol = m_ftol;
	lbfgs_param.wolfe = m_wolfe;
	lbfgs_param.gtol = m_gtol;
	lbfgs_param.xtol = m_xtol;
	lbfgs_param.orthantwise_c = m_orthantwise_c;
	lbfgs_param.orthantwise_start = m_orthantwise_start;
	lbfgs_param.orthantwise_end = m_orthantwise_end;

	float64_t nlml_opt=0;
	void * obj_prt = static_cast<void *>(this);

	Map<VectorXd> eigen_W(m_W.vector, m_W.vlen);
	lbfgs(m_W.vlen, m_W.vector, &nlml_opt,
		CKLDualInferenceMethod::evaluate,
		NULL, obj_prt, &lbfgs_param, CKLDualInferenceMethod::adjust_step);
	return nlml_opt;
}

SGVector<float64_t> CKLDualInferenceMethod::get_diagonal_vector()
{
	if (parameter_hash_changed())
		update();

	return SGVector<float64_t>(m_sW);
}

void CKLDualInferenceMethod::update_deriv()
{
	/* get_derivative_related_cov(MatrixXd eigen_dK) does the similar job
	 * Therefore, this function body is empty
	 */
}

void CKLDualInferenceMethod::update_chol()
{
	/* L is automatically updated when update_alpha is called
	 * Therefore, this function body is empty
	 */
}

void CKLDualInferenceMethod::update_approx_cov()
{
	m_Sigma=CMatrixOperations::get_inverse(m_L, m_ktrtr, m_sW, m_V, m_scale);
}

} /* namespace shogun */

#endif /* HAVE_EIGEN3 */
