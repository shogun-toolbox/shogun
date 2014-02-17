/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 *
 * Based on ideas from GAUSSIAN PROCESS REGRESSION AND CLASSIFICATION Toolbox
 * Copyright (C) 2005-2013 by Carl Edward Rasmussen & Hannes Nickisch under the
 * FreeBSD License
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 */

#include <shogun/machine/gp/EPInferenceMethod.h>

#ifdef HAVE_EIGEN3

#include <shogun/mathematics/Math.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/CombinedFeatures.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/lib/DynamicArray.h>

#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

// try to use previously allocated memory for SGVector
#define CREATE_SGVECTOR(vec, len, sg_type) \
	{ \
		if (!vec.vector || vec.vlen!=len) \
			vec=SGVector<sg_type>(len); \
	}

// try to use previously allocated memory for SGMatrix
#define CREATE_SGMATRIX(mat, rows, cols, sg_type) \
	{ \
		if (!mat.matrix || mat.num_rows!=rows || mat.num_cols!=cols) \
			mat=SGMatrix<sg_type>(rows, cols); \
	}

CEPInferenceMethod::CEPInferenceMethod()
{
	init();
}

CEPInferenceMethod::CEPInferenceMethod(CKernel* kernel, CFeatures* features,
		CMeanFunction* mean, CLabels* labels, CLikelihoodModel* model)
		: CInferenceMethod(kernel, features, mean, labels, model)
{
	init();
}

CEPInferenceMethod::~CEPInferenceMethod()
{
}

void CEPInferenceMethod::init()
{
	m_max_sweep=15;
	m_min_sweep=2;
	m_tol=1e-4;
}

float64_t CEPInferenceMethod::get_negative_log_marginal_likelihood()
{
	if (update_parameter_hash())
		update();

	return m_nlZ;
}

SGVector<float64_t> CEPInferenceMethod::get_alpha()
{
	if (update_parameter_hash())
		update();

	return SGVector<float64_t>(m_alpha);
}

SGMatrix<float64_t> CEPInferenceMethod::get_cholesky()
{
	if (update_parameter_hash())
		update();

	return SGMatrix<float64_t>(m_L);
}

SGVector<float64_t> CEPInferenceMethod::get_diagonal_vector()
{
	if (update_parameter_hash())
		update();

	return SGVector<float64_t>(m_sttau);
}

SGVector<float64_t> CEPInferenceMethod::get_posterior_mean()
{
	if (update_parameter_hash())
		update();

	return SGVector<float64_t>(m_mu);
}

SGMatrix<float64_t> CEPInferenceMethod::get_posterior_covariance()
{
	if (update_parameter_hash())
		update();

	return SGMatrix<float64_t>(m_Sigma);
}

void CEPInferenceMethod::update()
{
	// update kernel and feature matrix
	CInferenceMethod::update();

	// get number of labels (trainig examples)
	index_t n=m_labels->get_num_labels();

	// try to use tilde values from previous call
	if (m_ttau.vlen==n)
	{
		update_chol();
		update_approx_cov();
		update_approx_mean();
		update_negative_ml();
	}

	// get mean vector
	SGVector<float64_t> mean=m_mean->get_mean_vector(m_features);

	// get and scale diagonal of the kernel matrix
	SGVector<float64_t> ktrtr_diag=m_ktrtr.get_diagonal_vector();
	ktrtr_diag.scale(CMath::sq(m_scale));

	// marginal likelihood for ttau = tnu = 0
	float64_t nlZ0=-SGVector<float64_t>::sum(m_model->get_log_zeroth_moments(
			mean, ktrtr_diag, m_labels));

	// use zero values if we have no better guess or it's better
	if (m_ttau.vlen!=n || m_nlZ>nlZ0)
	{
		CREATE_SGVECTOR(m_ttau, n, float64_t);
		m_ttau.zero();

		CREATE_SGVECTOR(m_sttau, n, float64_t);
		m_sttau.zero();

		CREATE_SGVECTOR(m_tnu, n, float64_t);
		m_tnu.zero();

		CREATE_SGMATRIX(m_Sigma, m_ktrtr.num_rows, m_ktrtr.num_cols, float64_t);

		// copy data manually, since we don't have appropriate method
		for (index_t i=0; i<m_ktrtr.num_rows; i++)
			for (index_t j=0; j<m_ktrtr.num_cols; j++)
				m_Sigma(i,j)=m_ktrtr(i,j)*CMath::sq(m_scale);

		CREATE_SGVECTOR(m_mu, n, float64_t);
		m_mu.zero();

		// set marginal likelihood
		m_nlZ=nlZ0;
	}

	// create vector of the random permutation
	SGVector<index_t> randperm=SGVector<index_t>::randperm_vec(n);

	// cavity tau and nu vectors
	SGVector<float64_t> tau_n(n);
	SGVector<float64_t> nu_n(n);

	// cavity mu and s2 vectors
	SGVector<float64_t> mu_n(n);
	SGVector<float64_t> s2_n(n);

	float64_t nlZ_old=CMath::INFTY;
	uint32_t sweep=0;

	while ((CMath::abs(m_nlZ-nlZ_old)>m_tol && sweep<m_max_sweep) ||
			sweep<m_min_sweep)
	{
		nlZ_old=m_nlZ;
		sweep++;

		// shuffle random permutation
		randperm.permute();

		for (index_t j=0; j<n; j++)
		{
			index_t i=randperm[j];

			// find cavity paramters
			tau_n[i]=1.0/m_Sigma(i,i)-m_ttau[i];
			nu_n[i]=m_mu[i]/m_Sigma(i,i)+mean[i]*tau_n[i]-m_tnu[i];

			// compute cavity mean and variance
			mu_n[i]=nu_n[i]/tau_n[i];
			s2_n[i]=1.0/tau_n[i];

			// get moments
			float64_t mu=m_model->get_first_moment(mu_n, s2_n, m_labels, i);
			float64_t s2=m_model->get_second_moment(mu_n, s2_n, m_labels, i);

			// save old value of ttau
			float64_t ttau_old=m_ttau[i];

			// compute ttau and sqrt(ttau)
			m_ttau[i]=CMath::max(1.0/s2-tau_n[i], 0.0);
			m_sttau[i]=CMath::sqrt(m_ttau[i]);

			// compute tnu
			m_tnu[i]=mu/s2-nu_n[i];

			// compute difference ds2=ttau_new-ttau_old
			float64_t ds2=m_ttau[i]-ttau_old;

			// create eigen representation of Sigma, tnu and mu
			Map<MatrixXd> eigen_Sigma(m_Sigma.matrix, m_Sigma.num_rows,
					m_Sigma.num_cols);
			Map<VectorXd> eigen_tnu(m_tnu.vector, m_tnu.vlen);
			Map<VectorXd> eigen_mu(m_mu.vector, m_mu.vlen);

			VectorXd eigen_si=eigen_Sigma.col(i);

			// rank-1 update Sigma
			eigen_Sigma=eigen_Sigma-ds2/(1.0+ds2*eigen_si(i))*eigen_si*
				eigen_si.adjoint();

			// update mu
			eigen_mu=eigen_Sigma*eigen_tnu;
		}

		// update upper triangular factor (L^T) of Cholesky decomposition of
		// matrix B, approximate posterior covariance and mean, negative
		// marginal likelihood
		update_chol();
		update_approx_cov();
		update_approx_mean();
		update_negative_ml();
	}

	if (sweep==m_max_sweep && CMath::abs(m_nlZ-nlZ_old)>m_tol)
	{
		SG_ERROR("Maximum number (%d) of sweeps reached, but tolerance (%f) was "
				"not yet reached. You can manually set maximum number of sweeps "
				"or tolerance to fix this problem.\n", m_max_sweep, m_tol);
	}

	// update vector alpha
	update_alpha();

	// update matrices to compute derivatives
	update_deriv();
}

void CEPInferenceMethod::update_alpha()
{
	// create eigen representations kernel matrix, L^T, sqrt(ttau) and tnu
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<VectorXd> eigen_tnu(m_tnu.vector, m_tnu.vlen);
	Map<VectorXd> eigen_sttau(m_sttau.vector, m_sttau.vlen);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);

	// create shogun and eigen representation of the alpha vector
	CREATE_SGVECTOR(m_alpha, m_tnu.vlen, float64_t);
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);

	// solve LL^T * v = tS^(1/2) * K * tnu
	VectorXd eigen_v=eigen_L.triangularView<Upper>().adjoint().solve(
			eigen_sttau.cwiseProduct(eigen_K*CMath::sq(m_scale)*eigen_tnu));
	eigen_v=eigen_L.triangularView<Upper>().solve(eigen_v);

	// compute alpha = (I - tS^(1/2) * B^(-1) * tS(1/2) * K) * tnu =
	// tnu - tS(1/2) * (L^T)^(-1) * L^(-1) * tS^(1/2) * K * tnu =
	// tnu - tS(1/2) * v
	eigen_alpha=eigen_tnu-eigen_sttau.cwiseProduct(eigen_v);
}

void CEPInferenceMethod::update_chol()
{
	// create eigen representations of kernel matrix and sqrt(ttau)
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<VectorXd> eigen_sttau(m_sttau.vector, m_sttau.vlen);

	// create shogun and eigen representation of the upper triangular factor
	// (L^T) of the Cholesky decomposition of the matrix B
	CREATE_SGMATRIX(m_L, m_ktrtr.num_rows, m_ktrtr.num_cols, float64_t);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);

	// compute upper triangular factor L^T of the Cholesky decomposion of the
	// matrix: B = tS^(1/2) * K * tS^(1/2) + I
	LLT<MatrixXd> eigen_chol((eigen_sttau*eigen_sttau.adjoint()).cwiseProduct(
			eigen_K*CMath::sq(m_scale))+
			MatrixXd::Identity(m_L.num_rows, m_L.num_cols));

	eigen_L=eigen_chol.matrixU();
}

void CEPInferenceMethod::update_approx_cov()
{
	// create eigen representations of kernel matrix, L^T matrix and sqrt(ttau)
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<VectorXd> eigen_sttau(m_sttau.vector, m_sttau.vlen);

	// create shogun and eigen representation of the approximate covariance
	// matrix
	CREATE_SGMATRIX(m_Sigma, m_ktrtr.num_rows, m_ktrtr.num_cols, float64_t);
	Map<MatrixXd> eigen_Sigma(m_Sigma.matrix, m_Sigma.num_rows, m_Sigma.num_cols);

	// compute V = L^(-1) * tS^(1/2) * K, using upper triangular factor L^T
	MatrixXd eigen_V=eigen_L.triangularView<Upper>().adjoint().solve(
			eigen_sttau.asDiagonal()*eigen_K*CMath::sq(m_scale));

	// compute covariance matrix of the posterior:
	// Sigma = K - K * tS^(1/2) * (L * L^T)^(-1) * tS^(1/2) * K =
	// K - (K * tS^(1/2)) * (L^T)^(-1) * L^(-1) * tS^(1/2) * K =
	// K - (tS^(1/2) * K)^T * (L^(-1))^T * L^(-1) * tS^(1/2) * K = K - V^T * V
	eigen_Sigma=eigen_K*CMath::sq(m_scale)-eigen_V.adjoint()*eigen_V;
}

void CEPInferenceMethod::update_approx_mean()
{
	// create eigen representation of posterior covariance matrix and tnu
	Map<MatrixXd> eigen_Sigma(m_Sigma.matrix, m_Sigma.num_rows, m_Sigma.num_cols);
	Map<VectorXd> eigen_tnu(m_tnu.vector, m_tnu.vlen);

	// create shogun and eigen representation of the approximate mean vector
	CREATE_SGVECTOR(m_mu, m_tnu.vlen, float64_t);
	Map<VectorXd> eigen_mu(m_mu.vector, m_mu.vlen);

	// compute mean vector of the approximate posterior: mu = Sigma * tnu
	eigen_mu=eigen_Sigma*eigen_tnu;
}

void CEPInferenceMethod::update_negative_ml()
{
	// create eigen representation of Sigma, L, mu, tnu, ttau
	Map<MatrixXd> eigen_Sigma(m_Sigma.matrix, m_Sigma.num_rows, m_Sigma.num_cols);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);
	Map<VectorXd> eigen_mu(m_mu.vector, m_mu.vlen);
	Map<VectorXd> eigen_tnu(m_tnu.vector, m_tnu.vlen);
	Map<VectorXd> eigen_ttau(m_ttau.vector, m_ttau.vlen);

	// get and create eigen representation of the mean vector
	SGVector<float64_t> m=m_mean->get_mean_vector(m_features);
	Map<VectorXd> eigen_m(m.vector, m.vlen);

	// compute vector of cavity parameter tau_n
	VectorXd eigen_tau_n=(VectorXd::Ones(m_ttau.vlen)).cwiseQuotient(
			eigen_Sigma.diagonal())-eigen_ttau;

	// compute vector of cavity parameter nu_n
	VectorXd eigen_nu_n=eigen_mu.cwiseQuotient(eigen_Sigma.diagonal())-
		eigen_tnu+eigen_m.cwiseProduct(eigen_tau_n);

	// compute cavity mean: mu_n=nu_n/tau_n
	SGVector<float64_t> mu_n(m_ttau.vlen);
	Map<VectorXd> eigen_mu_n(mu_n.vector, mu_n.vlen);

	eigen_mu_n=eigen_nu_n.cwiseQuotient(eigen_tau_n);

	// compute cavity variance: s2_n=1.0/tau_n
	SGVector<float64_t> s2_n(m_ttau.vlen);
	Map<VectorXd> eigen_s2_n(s2_n.vector, s2_n.vlen);

	eigen_s2_n=(VectorXd::Ones(m_ttau.vlen)).cwiseQuotient(eigen_tau_n);

	float64_t lZ=SGVector<float64_t>::sum(
			m_model->get_log_zeroth_moments(mu_n, s2_n, m_labels));

	// compute nlZ_part1=sum(log(diag(L)))-sum(lZ)-tnu'*Sigma*tnu/2
	float64_t nlZ_part1=eigen_L.diagonal().array().log().sum()-lZ-
		(eigen_tnu.adjoint()*eigen_Sigma).dot(eigen_tnu)/2.0;

	// compute nlZ_part2=sum(tnu.^2./(tau_n+ttau))/2-sum(log(1+ttau./tau_n))/2
	float64_t nlZ_part2=(eigen_tnu.array().square()/
		(eigen_tau_n+eigen_ttau).array()).sum()/2.0-(1.0+eigen_ttau.array()/
		eigen_tau_n.array()).log().sum()/2.0;

	// compute nlZ_part3=-(nu_n-m.*tau_n)'*((ttau./tau_n.*(nu_n-m.*tau_n)-2*tnu)
	// ./(ttau+tau_n))/2
	float64_t nlZ_part3=-(eigen_nu_n-eigen_m.cwiseProduct(eigen_tau_n)).dot(
		((eigen_ttau.array()/eigen_tau_n.array()*(eigen_nu_n.array()-
		eigen_m.array()*eigen_tau_n.array())-2*eigen_tnu.array())/
		(eigen_ttau.array()+eigen_tau_n.array())).matrix())/2.0;

	// compute nlZ=nlZ_part1+nlZ_part2+nlZ_part3
	m_nlZ=nlZ_part1+nlZ_part2+nlZ_part3;
}

void CEPInferenceMethod::update_deriv()
{
	// create eigen representation of L, sstau, alpha
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);
	Map<VectorXd> eigen_sttau(m_sttau.vector, m_sttau.vlen);
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);

	// create shogun and eigen representation of F
	m_F=SGMatrix<float64_t>(m_L.num_rows, m_L.num_cols);
	Map<MatrixXd> eigen_F(m_F.matrix, m_F.num_rows, m_F.num_cols);

	// solve L*L^T * V = diag(sqrt(ttau))
	MatrixXd V=eigen_L.triangularView<Upper>().adjoint().solve(
			MatrixXd(eigen_sttau.asDiagonal()));
	V=eigen_L.triangularView<Upper>().solve(V);

	// compute F=alpha*alpha'-repmat(sW,1,n).*solve_chol(L,diag(sW))
	eigen_F=eigen_alpha*eigen_alpha.adjoint()-eigen_sttau.asDiagonal()*V;
}

SGVector<float64_t> CEPInferenceMethod::get_derivative_wrt_inference_method(
		const TParameter* param)
{
	REQUIRE(!strcmp(param->m_name, "scale"), "Can't compute derivative of "
			"the nagative log marginal likelihood wrt %s.%s parameter\n",
			get_name(), param->m_name)

	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<MatrixXd> eigen_F(m_F.matrix, m_F.num_rows, m_F.num_cols);

	SGVector<float64_t> result(1);

	// compute derivative wrt kernel scale: dnlZ=-sum(F.*K*scale*2)/2
	result[0]=-(eigen_F.cwiseProduct(eigen_K)*m_scale*2.0).sum()/2.0;

	return result;
}

SGVector<float64_t> CEPInferenceMethod::get_derivative_wrt_likelihood_model(
		const TParameter* param)
{
	SG_NOTIMPLEMENTED
	return SGVector<float64_t>();
}

SGVector<float64_t> CEPInferenceMethod::get_derivative_wrt_kernel(
		const TParameter* param)
{
	// create eigen representation of the matrix Q
	Map<MatrixXd> eigen_F(m_F.matrix, m_F.num_rows, m_F.num_cols);

	SGVector<float64_t> result;

	if (param->m_datatype.m_ctype==CT_VECTOR ||
			param->m_datatype.m_ctype==CT_SGVECTOR)
	{
		REQUIRE(param->m_datatype.m_length_y,
				"Length of the parameter %s should not be NULL\n", param->m_name)
		result=SGVector<float64_t>(*(param->m_datatype.m_length_y));
	}
	else
	{
		result=SGVector<float64_t>(1);
	}

	for (index_t i=0; i<result.vlen; i++)
	{
		SGMatrix<float64_t> dK;

		if (result.vlen==1)
			dK=m_kernel->get_parameter_gradient(param);
		else
			dK=m_kernel->get_parameter_gradient(param, i);

		Map<MatrixXd> eigen_dK(dK.matrix, dK.num_rows, dK.num_cols);

		// compute derivative wrt kernel parameter: dnlZ=-sum(F.*dK*scale^2)/2.0
		result[i]=-(eigen_F.cwiseProduct(eigen_dK)*CMath::sq(m_scale)).sum()/2.0;
	}

	return result;
}

SGVector<float64_t> CEPInferenceMethod::get_derivative_wrt_mean(
		const TParameter* param)
{
	SG_NOTIMPLEMENTED
	return SGVector<float64_t>();
}

#endif /* HAVE_EIGEN3 */
