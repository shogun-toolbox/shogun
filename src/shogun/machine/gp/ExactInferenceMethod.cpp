/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 * Copyright (C) 2012 Jacob Walker
 * Copyright (C) 2013 Roman Votyakov
 *
 * Code adapted from Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 */

#include <shogun/machine/gp/ExactInferenceMethod.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CExactInferenceMethod::CExactInferenceMethod() : CInferenceMethod()
{
}

CExactInferenceMethod::CExactInferenceMethod(CKernel* kern, CFeatures* feat,
		CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod) :
		CInferenceMethod(kern, feat, m, lab, mod)
{
}

CExactInferenceMethod::~CExactInferenceMethod()
{
}

void CExactInferenceMethod::update()
{
	SG_DEBUG("entering\n");

	CInferenceMethod::update();
	update_chol();
	update_alpha();
	if (m_is_compute_gradients)
	{
		update_deriv();
		update_mean();
		update_cov();
	}
	update_parameter_hash();

	SG_DEBUG("leaving\n");
}

void CExactInferenceMethod::check_members() const
{
	CInferenceMethod::check_members();

	REQUIRE(m_model->get_model_type()==LT_GAUSSIAN,
		"Exact inference method can only use Gaussian likelihood function\n")
	REQUIRE(m_labels->get_label_type()==LT_REGRESSION,
		"Labels must be type of CRegressionLabels\n")
}

SGVector<float64_t> CExactInferenceMethod::get_diagonal_vector()
{
	if (parameter_hash_changed())
		update();

	// get the sigma variable from the Gaussian likelihood model
	CGaussianLikelihood* lik=CGaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);

	// compute diagonal vector: sW=1/sigma
	SGVector<float64_t> result(m_features->get_num_vectors());
	result.fill_vector(result.vector, m_features->get_num_vectors(), 1.0/sigma);

	return result;
}

float64_t CExactInferenceMethod::get_negative_log_marginal_likelihood()
{
	if (parameter_hash_changed())
		update();

	// get the sigma variable from the Gaussian likelihood model
	CGaussianLikelihood* lik=CGaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);

	// create eigen representation of alpha and L
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);

	// get labels and mean vectors and create eigen representation
	SGVector<float64_t> y=((CRegressionLabels*) m_labels)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);
	SGVector<float64_t> m=m_mean->get_mean_vector(m_features);
	Map<VectorXd> eigen_m(m.vector, m.vlen);

	// compute negative log of the marginal likelihood:
	// nlZ=(y-m)'*alpha/2+sum(log(diag(L)))+n*log(2*pi*sigma^2)/2
	float64_t result=(eigen_y-eigen_m).dot(eigen_alpha)/2.0+
		eigen_L.diagonal().array().log().sum()+m_L.num_rows*
		CMath::log(2*CMath::PI*CMath::sq(sigma))/2.0;

	return result;
}

SGVector<float64_t> CExactInferenceMethod::get_alpha()
{
	if (parameter_hash_changed())
		update();

	return SGVector<float64_t>(m_alpha);
}

SGMatrix<float64_t> CExactInferenceMethod::get_cholesky()
{
	if (parameter_hash_changed())
		update();

	return SGMatrix<float64_t>(m_L);
}

SGVector<float64_t> CExactInferenceMethod::get_posterior_mean()
{
	check_compute_gradients();
	if (parameter_hash_changed())
		update();

	return SGVector<float64_t>(m_mu);
}

SGMatrix<float64_t> CExactInferenceMethod::get_posterior_covariance()
{
	check_compute_gradients();

	if (parameter_hash_changed())
		update();

	return SGMatrix<float64_t>(m_Sigma);
}

void CExactInferenceMethod::update_chol()
{
	// get the sigma variable from the Gaussian likelihood model
	CGaussianLikelihood* lik=CGaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);

	/* check whether to allocate cholesky memory */
	if (!m_L.matrix || m_L.num_rows!=m_ktrtr.num_rows)
		m_L=SGMatrix<float64_t>(m_ktrtr.num_rows, m_ktrtr.num_cols);

	/* creates views on kernel and cholesky matrix and perform cholesky */
	Map<MatrixXd> K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<MatrixXd> L(m_L.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	LLT<MatrixXd> llt(K*(CMath::sq(m_scale)/CMath::sq(sigma))+
		MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols));
	L=llt.matrixU();
}

void CExactInferenceMethod::update_alpha()
{
	// get the sigma variable from the Gaussian likelihood model
	CGaussianLikelihood* lik=CGaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);

	// get labels and mean vector and create eigen representation
	SGVector<float64_t> y=((CRegressionLabels*) m_labels)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);
	SGVector<float64_t> m=m_mean->get_mean_vector(m_features);
	Map<VectorXd> eigen_m(m.vector, m.vlen);

	m_alpha=SGVector<float64_t>(y.vlen);

	/* creates views on cholesky matrix and alpha and solve system
	 * (L * L^T) * a = y for a */
	Map<VectorXd> a(m_alpha.vector, m_alpha.vlen);
	Map<MatrixXd> L(m_L.matrix, m_L.num_rows, m_L.num_cols);

	a=L.triangularView<Upper>().adjoint().solve(eigen_y-eigen_m);
	a=L.triangularView<Upper>().solve(a);

	a/=CMath::sq(sigma);
}

void CExactInferenceMethod::update_mean()
{
	// create eigen representataion of kernel matrix and alpha
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);

	// get mean and create eigen representation of it
	SGVector<float64_t> m=m_mean->get_mean_vector(m_features);
	Map<VectorXd> eigen_m(m.vector, m.vlen);

	m_mu=SGVector<float64_t>(m.vlen);
	Map<VectorXd> eigen_mu(m_mu.vector, m_mu.vlen);

	eigen_mu=eigen_K*CMath::sq(m_scale)*eigen_alpha;
}

void CExactInferenceMethod::update_cov()
{
	// create eigen representataion of upper triangular factor L^T and kernel
	// matrix
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);

	m_Sigma=SGMatrix<float64_t>(m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<MatrixXd> eigen_Sigma(m_Sigma.matrix, m_Sigma.num_rows,
			m_Sigma.num_cols);

	// compute V = L^(-1) * K, using upper triangular factor L^T
	MatrixXd eigen_V=eigen_L.triangularView<Upper>().adjoint().solve(
			eigen_K*CMath::sq(m_scale));

	CGaussianLikelihood* lik=CGaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);
	eigen_V = eigen_V/sigma;

	// compute covariance matrix of the posterior: Sigma = K - V^T * V
	eigen_Sigma=eigen_K*CMath::sq(m_scale)-eigen_V.adjoint()*eigen_V;
}

void CExactInferenceMethod::update_deriv()
{
	// get the sigma variable from the Gaussian likelihood model
	CGaussianLikelihood* lik=CGaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);

	// create eigen representation of derivative matrix and cholesky
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);

	m_Q=SGMatrix<float64_t>(m_L.num_rows, m_L.num_cols);
	Map<MatrixXd> eigen_Q(m_Q.matrix, m_Q.num_rows, m_Q.num_cols);

	// solve L * L' * Q = I
	eigen_Q=eigen_L.triangularView<Upper>().adjoint().solve(
		MatrixXd::Identity(m_L.num_rows, m_L.num_cols));
	eigen_Q=eigen_L.triangularView<Upper>().solve(eigen_Q);

	// divide Q by sigma^2
	eigen_Q/=CMath::sq(sigma);

	// create eigen representation of alpha and compute Q=Q-alpha*alpha'
	eigen_Q-=eigen_alpha*eigen_alpha.transpose();
}

SGVector<float64_t> CExactInferenceMethod::get_derivative_wrt_inference_method(
		const TParameter* param)
{
	REQUIRE(!strcmp(param->m_name, "scale"), "Can't compute derivative of "
			"the nagative log marginal likelihood wrt %s.%s parameter\n",
			get_name(), param->m_name)

	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<MatrixXd> eigen_Q(m_Q.matrix, m_Q.num_rows, m_Q.num_cols);

	SGVector<float64_t> result(1);

	// compute derivative wrt kernel scale: dnlZ=sum(Q.*K*scale*2)/2
	result[0]=(eigen_Q.cwiseProduct(eigen_K)*m_scale*2.0).sum()/2.0;

	return result;
}

CExactInferenceMethod* CExactInferenceMethod::obtain_from_generic(
		CInferenceMethod* inference)
{
	if (inference==NULL)
		return NULL;

	if (inference->get_inference_type()!=INF_EXACT)
		SG_SERROR("Provided inference is not of type CExactInferenceMethod!\n")

	SG_REF(inference);
	return (CExactInferenceMethod*)inference;
}

SGVector<float64_t> CExactInferenceMethod::get_derivative_wrt_likelihood_model(
		const TParameter* param)
{
	REQUIRE(!strcmp(param->m_name, "sigma"), "Can't compute derivative of "
			"the nagative log marginal likelihood wrt %s.%s parameter\n",
			m_model->get_name(), param->m_name)

	// get the sigma variable from the Gaussian likelihood model
	CGaussianLikelihood* lik=CGaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);

	// create eigen representation of the matrix Q
	Map<MatrixXd> eigen_Q(m_Q.matrix, m_Q.num_rows, m_Q.num_cols);

	SGVector<float64_t> result(1);

	// compute derivative wrt likelihood model parameter sigma:
	// dnlZ=sigma^2*trace(Q)
	result[0]=CMath::sq(sigma)*eigen_Q.trace();

	return result;
}

SGVector<float64_t> CExactInferenceMethod::get_derivative_wrt_kernel(
		const TParameter* param)
{
	// create eigen representation of the matrix Q
	Map<MatrixXd> eigen_Q(m_Q.matrix, m_Q.num_rows, m_Q.num_cols);

	REQUIRE(param, "Param not set\n");
	SGVector<float64_t> result;
	int64_t len=const_cast<TParameter *>(param)->m_datatype.get_num_elements();
	result=SGVector<float64_t>(len);

	for (index_t i=0; i<result.vlen; i++)
	{
		SGMatrix<float64_t> dK;

		if (result.vlen==1)
			dK=m_kernel->get_parameter_gradient(param);
		else
			dK=m_kernel->get_parameter_gradient(param, i);

		Map<MatrixXd> eigen_dK(dK.matrix, dK.num_rows, dK.num_cols);

		// compute derivative wrt kernel parameter: dnlZ=sum(Q.*dK*scale)/2.0
		result[i]=(eigen_Q.cwiseProduct(eigen_dK)*CMath::sq(m_scale)).sum()/2.0;
	}

	return result;
}

SGVector<float64_t> CExactInferenceMethod::get_derivative_wrt_mean(
		const TParameter* param)
{
	// create eigen representation of alpha vector
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);

	REQUIRE(param, "Param not set\n");
	SGVector<float64_t> result;
	int64_t len=const_cast<TParameter *>(param)->m_datatype.get_num_elements();
	result=SGVector<float64_t>(len);

	for (index_t i=0; i<result.vlen; i++)
	{
		SGVector<float64_t> dmu;

		if (result.vlen==1)
			dmu=m_mean->get_parameter_derivative(m_features, param);
		else
			dmu=m_mean->get_parameter_derivative(m_features, param, i);

		Map<VectorXd> eigen_dmu(dmu.vector, dmu.vlen);

		// compute derivative wrt mean parameter: dnlZ=-dmu'*alpha
		result[i]=-eigen_dmu.dot(eigen_alpha);
	}

	return result;
}

#endif /* HAVE_EIGEN3 */
