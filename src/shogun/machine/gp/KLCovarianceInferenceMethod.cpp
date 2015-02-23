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
 * Nickisch, Hannes, and Carl Edward Rasmussen.
 * "Approximations for Binary Gaussian Process Classification."
 * Journal of Machine Learning Research 9.10 (2008).
 *
 * This code specifically adapted from function in approxKL.m and infKL.m
 */

#include <shogun/machine/gp/KLCovarianceInferenceMethod.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Math.h>
#include <shogun/machine/gp/MatrixOperations.h>
#include <shogun/machine/gp/VariationalGaussianLikelihood.h>

using namespace Eigen;

namespace shogun
{

CKLCovarianceInferenceMethod::CKLCovarianceInferenceMethod() : CKLInferenceMethod()
{
	init();
}

CKLCovarianceInferenceMethod::CKLCovarianceInferenceMethod(CKernel* kern,
		CFeatures* feat, CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod)
		: CKLInferenceMethod(kern, feat, m, lab, mod)
{
	init();
}

void CKLCovarianceInferenceMethod::init()
{
	SG_ADD(&m_V, "V",
		"V is L'*V=diag(sW)*K",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_A, "A",
		"A is A=I-K*diag(sW)*inv(L)'*inv(L)*diag(sW)",
		MS_NOT_AVAILABLE);
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
}


SGVector<float64_t> CKLCovarianceInferenceMethod::get_alpha()
{
	/** Note that m_alpha contains not only the alpha vector defined in the reference
	 * but also a vector corresponding to the diagonal part of W 
	 *
	 * Note that alpha=K^{-1}(mu-mean), where mean is generated from mean function,
	 * K is generated from cov function
	 * and mu is not only the posterior mean but also the variational mean
	 */
	if (parameter_hash_changed())
		update();

	index_t len=m_alpha.vlen/2;
	SGVector<float64_t> result(len);

	Map<VectorXd> eigen_result(result.vector, len);
	Map<VectorXd> eigen_alpha(m_alpha.vector, len);

	eigen_result=eigen_alpha;

	return result;
}

CKLCovarianceInferenceMethod::~CKLCovarianceInferenceMethod()
{
}

CKLCovarianceInferenceMethod* CKLCovarianceInferenceMethod::obtain_from_generic(
		CInferenceMethod* inference)
{
	REQUIRE(inference, "Inference pointer not set.\n");

	if (inference->get_inference_type()!=INF_KL_COVARIANCE)
		SG_SERROR("Provided inference is not of type CKLCovarianceInferenceMethod!\n")

	SG_REF(inference);
	return (CKLCovarianceInferenceMethod*)inference;
}

bool CKLCovarianceInferenceMethod::lbfgs_precompute()
{
	SGVector<float64_t> mean=m_mean->get_mean_vector(m_features);
	Map<VectorXd> eigen_mean(mean.vector, mean.vlen);
	
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);

	index_t len=m_alpha.vlen/2;
	//construct mu
	Map<VectorXd> eigen_alpha(m_alpha.vector, len);

	Map<VectorXd> eigen_mu(m_mu.vector, m_mu.vlen);
	//mu=K*alpha+m
	eigen_mu=eigen_K*CMath::sq(m_scale)*eigen_alpha+eigen_mean;

	//construct s2
	Map<VectorXd> eigen_log_neg_lambda(m_alpha.vector+len, len);

	Map<VectorXd> eigen_W(m_W.vector, m_W.vlen);
	eigen_W=(2.0*eigen_log_neg_lambda.array().exp()).matrix();

	Map<VectorXd> eigen_sW(m_sW.vector, m_sW.vlen);
	eigen_sW=eigen_W.array().sqrt().matrix();

	m_L=CMatrixOperations::get_choleksy(m_W, m_sW, m_ktrtr, m_scale);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);
	
	//solve L'*V=diag(sW)*K
	Map<MatrixXd> eigen_V(m_V.matrix, m_V.num_rows, m_V.num_cols);
	eigen_V=eigen_L.triangularView<Upper>().adjoint().solve(eigen_sW.asDiagonal()*eigen_K*CMath::sq(m_scale));
	Map<VectorXd> eigen_s2(m_s2.vector, m_s2.vlen);
	//Sigma=inv(inv(K)-2*diag(lambda))=K-K*diag(sW)*inv(L)'*inv(L)*diag(sW)*K
	//v=abs(diag(Sigma))
	eigen_s2=(eigen_K.diagonal().array()*CMath::sq(m_scale)-(eigen_V.array().pow(2).colwise().sum().transpose())).abs().matrix();

	CVariationalGaussianLikelihood * lik=get_variational_likelihood();
	bool status = lik->set_variational_distribution(m_mu, m_s2, m_labels);
	return status;
}

void CKLCovarianceInferenceMethod::get_gradient_of_nlml_wrt_parameters(SGVector<float64_t> gradient)
{
	REQUIRE(gradient.vlen==m_alpha.vlen,
		"The length of gradients (%d) should the same as the length of parameters (%d)\n",
		gradient.vlen, m_alpha.vlen);

	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);
	Map<VectorXd> eigen_sW(m_sW.vector, m_sW.vlen);
	Map<MatrixXd> eigen_V(m_V.matrix, m_V.num_rows, m_V.num_cols);
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<VectorXd> eigen_s2(m_s2.vector, m_s2.vlen);

	index_t len=m_alpha.vlen/2;
	Map<VectorXd> eigen_alpha(m_alpha.vector, len);
	Map<VectorXd> eigen_log_neg_lambda(m_alpha.vector+len, len);

	CVariationalGaussianLikelihood * lik=get_variational_likelihood();
	lik->set_variational_distribution(m_mu, m_s2, m_labels);

	//[a,df,dV] = a_related2(mu,s2,y,lik);
	TParameter* s2_param=lik->m_parameters->get_parameter("sigma2");
	m_dv=lik->get_variational_first_derivative(s2_param);
	Map<VectorXd> eigen_dv(m_dv.vector, m_dv.vlen);

	TParameter* mu_param=lik->m_parameters->get_parameter("mu");
	m_df=lik->get_variational_first_derivative(mu_param);
	Map<VectorXd> eigen_df(m_df.vector, m_df.vlen);
	//U=inv(L')*diag(sW)
	MatrixXd eigen_U=eigen_L.triangularView<Upper>().adjoint().solve(MatrixXd(eigen_sW.asDiagonal()));
	Map<MatrixXd> eigen_A(m_A.matrix, m_A.num_rows, m_A.num_cols);
	// A=I-K*diag(sW)*inv(L)*inv(L')*diag(sW)
	eigen_A=MatrixXd::Identity(len, len)-eigen_V.transpose()*eigen_U;

	m_Sigma=CMatrixOperations::get_inverse(m_L, m_ktrtr, m_sW, m_V, m_scale);
	Map<MatrixXd> eigen_Sigma(m_Sigma.matrix, m_Sigma.num_rows, m_Sigma.num_cols);

	Map<VectorXd> eigen_dnlz_alpha(gradient.vector, len);
	Map<VectorXd> eigen_dnlz_log_neg_lambda(gradient.vector+len, len);

	//dlZ_alpha  = K*(df-alpha);
	eigen_dnlz_alpha=eigen_K*CMath::sq(m_scale)*(-eigen_df+eigen_alpha);

	//dlZ_lambda = 2*(Sigma.*Sigma)*dV +v -sum(Sigma.*A,2);   % => fast diag(V*VinvK')
	//dlZ_log_neg_lambda = dlZ_lambda .* lambda;
	//dnlZ = -[dlZ_alpha; dlZ_log_neg_lambda];
	eigen_dnlz_log_neg_lambda=(eigen_Sigma.array().pow(2)*2.0).matrix()*eigen_dv+eigen_s2;
	eigen_dnlz_log_neg_lambda=eigen_dnlz_log_neg_lambda-(eigen_Sigma.array()*eigen_A.array()).rowwise().sum().matrix();
	eigen_dnlz_log_neg_lambda=(eigen_log_neg_lambda.array().exp()*eigen_dnlz_log_neg_lambda.array()).matrix();
}


float64_t CKLCovarianceInferenceMethod::get_negative_log_marginal_likelihood_helper()
{
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen/2);
	Map<VectorXd> eigen_mu(m_mu.vector, m_mu.vlen);
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	//get mean vector and create eigen representation of it
	SGVector<float64_t> mean=m_mean->get_mean_vector(m_features);
	Map<VectorXd> eigen_mean(mean.vector, mean.vlen);

	CVariationalGaussianLikelihood * lik=get_variational_likelihood();
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

float64_t CKLCovarianceInferenceMethod::get_derivative_related_cov(SGMatrix<float64_t> dK)
{
	Map<MatrixXd> eigen_dK(dK.matrix, dK.num_rows, dK.num_cols);
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<VectorXd> eigen_W(m_W.vector, m_W.vlen);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);
	Map<VectorXd> eigen_sW(m_sW.vector, m_sW.vlen);
	Map<MatrixXd> eigen_Sigma(m_Sigma.matrix, m_Sigma.num_rows, m_Sigma.num_cols);
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen/2);
	Map<MatrixXd> eigen_A(m_A.matrix, m_A.num_rows, m_A.num_cols);

	Map<VectorXd> eigen_dv(m_dv.vector, m_dv.vlen);
	Map<VectorXd> eigen_df(m_df.vector, m_df.vlen);

	//AdK = A*dK;
	MatrixXd AdK=eigen_A*eigen_dK;

	//z = diag(AdK) + sum(A.*AdK,2) - sum(A'.*AdK,1)';
	VectorXd z=AdK.diagonal()+(eigen_A.array()*AdK.array()).rowwise().sum().matrix()
		-(eigen_A.transpose().array()*AdK.array()).colwise().sum().transpose().matrix();

	//dnlZ(j) = alpha'*dK*(alpha/2-df) - z'*dv;
	return eigen_alpha.dot(eigen_dK*(eigen_alpha/2.0-eigen_df))-z.dot(eigen_dv);
}

void CKLCovarianceInferenceMethod::update_alpha()
{
	float64_t nlml_new=0;
	float64_t nlml_def=0;

	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);

	if (m_alpha.vlen == m_labels->get_num_labels()*2)
	{
		nlml_new=get_negative_log_marginal_likelihood_helper();

		float64_t trace=0;
		LLT<MatrixXd> llt((eigen_K*CMath::sq(m_scale))+
			MatrixXd::Identity(eigen_K.rows(), eigen_K.cols()));
		MatrixXd LL=llt.matrixU();
		MatrixXd tt=LL.triangularView<Upper>().adjoint().solve(MatrixXd::Identity(LL.rows(),LL.cols()));

		for(index_t idx=0; idx<tt.rows(); idx++)
			trace+=(tt.col(idx).array().pow(2)).sum();

		MatrixXd eigen_V=LL.triangularView<Upper>().adjoint().solve(eigen_K*CMath::sq(m_scale));
		SGVector<float64_t> s2_tmp(m_s2.vlen);
		Map<VectorXd> eigen_s2(s2_tmp.vector, s2_tmp.vlen);
		eigen_s2=(eigen_K.diagonal().array()*CMath::sq(m_scale)-(eigen_V.array().pow(2).colwise().sum().transpose())).abs().matrix();
		SGVector<float64_t> mean=m_mean->get_mean_vector(m_features);

		CVariationalGaussianLikelihood * lik=get_variational_likelihood();
		lik->set_variational_distribution(mean, s2_tmp, m_labels);
		float64_t a=SGVector<float64_t>::sum(lik->get_variational_expection());

		nlml_def=-a+LL.diagonal().array().log().sum();
		nlml_def+=0.5*(-eigen_K.rows()+trace);

		if (nlml_new<=nlml_def)
			lik->set_variational_distribution(m_mu, m_s2, m_labels);
	}

	if (m_alpha.vlen != m_labels->get_num_labels()*2 || nlml_def<nlml_new)
	{
		if(m_alpha.vlen != m_labels->get_num_labels()*2)
			m_alpha = SGVector<float64_t>(m_labels->get_num_labels()*2);

		//init
		for (index_t i=0; i<m_alpha.vlen; i++)
		{
			if (i<m_alpha.vlen/2)
				m_alpha[i]=0;
			else
				m_alpha[i]=CMath::log(0.5);
		}

		index_t len=m_alpha.vlen/2;
		m_W=SGVector<float64_t>(len);
		m_sW=SGVector<float64_t>(len);
		m_mu=SGVector<float64_t>(len);
		m_s2=SGVector<float64_t>(len);
		m_V=SGMatrix<float64_t>(len, len);
		m_Sigma=SGMatrix<float64_t>(len, len);
		m_A=SGMatrix<float64_t>(len, len);
	}

	nlml_new=lbfgs_optimization();
}

SGVector<float64_t> CKLCovarianceInferenceMethod::get_diagonal_vector()
{
	if (parameter_hash_changed())
		update();

	return SGVector<float64_t>(m_sW);
}

void CKLCovarianceInferenceMethod::update_deriv()
{
	/** get_derivative_related_cov() does the similar job
	 * Therefore, this function body is empty
	 */
}

void CKLCovarianceInferenceMethod::update_chol()
{
	/** L is automatically updated when update_alpha is called
	 * Therefore, this function body is empty
	 */
}

void CKLCovarianceInferenceMethod::update_approx_cov()
{
	/** The variational co-variational matrix,
	 * which is automatically computed when update_alpha is called,
	 * is an approximated posterior co-variance matrix
	 * Therefore, this function body is empty
	 */
}

} /* namespace shogun */

#endif /* HAVE_EIGEN3 */
