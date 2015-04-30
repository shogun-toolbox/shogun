/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <shogun/lib/config.h>

#ifdef HAVE_LAPACK

#include <vector>
#include <limits>
#include <algorithm>

#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/regression/LeastAngleRegression.h>
#include <shogun/labels/RegressionLabels.h>

using namespace shogun;
using namespace std;

static vector<float64_t> make_vector(int32_t size, float64_t val)
{
	vector<float64_t> result(size);
	fill(result.begin(), result.end(), val);
	return result;
}

static void plane_rot(float64_t x0, float64_t x1,
	float64_t &y0, float64_t &y1, SGMatrix<float64_t> &G)
{
	G.zero();
	if (x1 == 0)
	{
		G(0, 0) = G(1, 1) = 1;
		y0 = x0;
		y1 = x1;
	}
	else
	{
		float64_t r = CMath::sqrt(x0*x0+x1*x1);
		float64_t sx0 = x0 / r;
		float64_t sx1 = x1 / r;

		G(0,0) = sx0;
		G(1,0) = -sx1;
		G(0,1) = sx1;
		G(1,1) = sx0;

		y0 = r;
		y1 = 0;
	}
}

static void find_max_abs(const vector<float64_t> &vec, const vector<bool> &ignore_mask,
	int32_t &imax, float64_t& vmax)
{
	imax = -1;
	vmax = -1;
	for (uint32_t i=0; i < vec.size(); ++i)
	{
		if (ignore_mask[i])
			continue;

		if (CMath::abs(vec[i]) > vmax)
		{
			vmax = CMath::abs(vec[i]);
			imax = i;
		}
	}
}

CLeastAngleRegression::CLeastAngleRegression(bool lasso) :
	CLinearMachine(), m_lasso(lasso),
	m_max_nonz(0), m_max_l1_norm(0)
{
	SG_ADD(&m_max_nonz, "max_nonz", "Max number of non-zero variables", MS_AVAILABLE);
	SG_ADD(&m_max_l1_norm, "max_l1_norm", "Max l1-norm of estimator", MS_AVAILABLE);
}

CLeastAngleRegression::~CLeastAngleRegression()
{

}

bool CLeastAngleRegression::train_machine(CFeatures* data)
{
	if (!m_labels)
		SG_ERROR("No labels set\n")
	if (m_labels->get_label_type() != LT_REGRESSION)
		SG_ERROR("Expected RegressionLabels\n")

	if (!data)
		data=features;

	if (!data)
		SG_ERROR("No features set\n")

	if (m_labels->get_num_labels() != data->get_num_vectors())
		SG_ERROR("Number of training vectors does not match number of labels\n")

	if (data->get_feature_class() != C_DENSE)
		SG_ERROR("Expected Simple Features\n")

	if (data->get_feature_type() != F_DREAL)
		SG_ERROR("Expected Real Features\n")

	CDenseFeatures<float64_t>* feats=(CDenseFeatures<float64_t>*) data;
	int32_t n_fea = feats->get_num_features();
	int32_t n_vec = feats->get_num_vectors();

	bool lasso_cond = false;
	bool stop_cond = false;

	// init facilities
	m_beta_idx.clear();
	m_beta_path.clear();
	m_num_active = 0;
	m_active_set.clear();
	m_is_active.resize(n_fea);
	fill(m_is_active.begin(), m_is_active.end(), false);

	SGVector<float64_t> y = ((CRegressionLabels*) m_labels)->get_labels();
	SGMatrix<float64_t> Xorig = feats->get_feature_matrix();

	// transpose(X) is more convenient to work with since we care
	// about features here. After transpose, each row will be a data
	// point while each column corresponds to a feature
	SGMatrix<float64_t> X(n_vec, n_fea, true);
	for (int32_t i=0; i < n_vec; ++i)
	{
		for (int32_t j=0; j < n_fea; ++j)
			X(i,j) = Xorig(j,i);
	}

	// beta is the estimator
	vector<float64_t> beta = make_vector(n_fea, 0);

	vector<float64_t> Xy = make_vector(n_fea, 0);
	// Xy = X' * y
	cblas_dgemv(CblasColMajor, CblasTrans, n_vec, n_fea, 1, X.matrix, n_vec,
		y.vector, 1, 0, &Xy[0], 1);

	// mu is the prediction
	vector<float64_t> mu = make_vector(n_vec, 0);

	// correlation
	vector<float64_t> corr = make_vector(n_fea, 0);
	// sign of correlation
	vector<float64_t> corr_sign(n_fea);

	// Cholesky factorization R'R = X'X, R is upper triangular
	SGMatrix<float64_t> R;

	float64_t max_corr = 1;
	int32_t i_max_corr = 1;

	// first entry: all coefficients are zero
	m_beta_path.push_back(beta);
	m_beta_idx.push_back(0);

	//maximum allowed active variables at a time
	int32_t max_active_allowed = CMath::min(n_vec-1, n_fea);

	//========================================
	// main loop
	//========================================
	int32_t nloop=0;
	while (m_num_active < max_active_allowed && max_corr > CMath::MACHINE_EPSILON && !stop_cond)
	{
		// corr = X' * (y-mu) = - X'*mu + Xy
		copy(Xy.begin(), Xy.end(), corr.begin());
		cblas_dgemv(CblasColMajor, CblasTrans, n_vec, n_fea, -1,
			X.matrix, n_vec, &mu[0], 1, 1, &corr[0], 1);

		// corr_sign = sign(corr)
		for (uint32_t i=0; i < corr.size(); ++i)
			corr_sign[i] = CMath::sign(corr[i]);

		// find max absolute correlation in inactive set
		find_max_abs(corr, m_is_active, i_max_corr, max_corr);

		if (!lasso_cond)
		{
			// update Cholesky factorization matrix
			R=cholesky_insert(X, R, i_max_corr);
			activate_variable(i_max_corr);
		}

		// corr_sign_a = corr_sign[m_active_set]
		vector<float64_t> corr_sign_a(m_num_active);
		for (int32_t i=0; i < m_num_active; ++i)
			corr_sign_a[i] = corr_sign[m_active_set[i]];

		// GA1 = R\(R'\corr_sign_a)
		vector<float64_t> GA1(corr_sign_a);
		cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
			m_num_active, 1, 1, R.matrix, m_num_active, &GA1[0], m_num_active);
		cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
			m_num_active, 1, 1, R.matrix, m_num_active, &GA1[0], m_num_active);

		// AA = 1/sqrt(GA1' * corr_sign_a)
		float64_t AA = cblas_ddot(m_num_active, &GA1[0], 1, &corr_sign_a[0], 1);
		AA = 1/CMath::sqrt(AA);

		// wA = AA*GA1
		vector<float64_t> wA(GA1);
		for (int32_t i=0; i < m_num_active; ++i)
			wA[i] *= AA;

		// equiangular direction (unit vector)
		vector<float64_t> u = make_vector(n_vec, 0);
		// u = X[:,m_active_set] * wA
		for (int32_t i=0; i < m_num_active; ++i)
		{
			// u += wA[i] * X[:,m_active_set[i]]
			cblas_daxpy(n_vec, wA[i],
				X.get_column_vector(m_active_set[i]), 1, &u[0], 1);
		}

		// step size
		float64_t gamma = max_corr / AA;
		if (m_num_active < n_fea)
		{
			for (int32_t i=0; i < n_fea; ++i)
			{
				if (m_is_active[i])
					continue;

				// correlation between X[:,i] and u
				float64_t dir_corr = cblas_ddot(n_vec,
					X.get_column_vector(i), 1, &u[0], 1);

				float64_t tmp1 = (max_corr-corr[i])/(AA-dir_corr);
				float64_t tmp2 = (max_corr+corr[i])/(AA+dir_corr);
				if (tmp1 > CMath::MACHINE_EPSILON && tmp1 < gamma)
					gamma = tmp1;
				if (tmp2 > CMath::MACHINE_EPSILON && tmp2 < gamma)
					gamma = tmp2;
			}
		}

		int32_t i_kick=-1;
		int32_t i_change=i_max_corr;
		if (m_lasso)
		{
			// lasso modification to further refine gamma
			lasso_cond = false;
			float64_t lasso_bound = numeric_limits<float64_t>::max();

			for (int32_t i=0; i < m_num_active; ++i)
			{
				float64_t tmp = -beta[m_active_set[i]] / wA[i];
				if (tmp > CMath::MACHINE_EPSILON && tmp < lasso_bound)
				{
					lasso_bound = tmp;
					i_kick = i;
				}
			}

			if (lasso_bound < gamma)
			{
				gamma = lasso_bound;
				lasso_cond = true;
				i_change = m_active_set[i_kick];
			}
		}

		// update prediction: mu = mu + gamma * u
		cblas_daxpy(n_vec, gamma, &u[0], 1, &mu[0], 1);

		// update estimator
		for (int32_t i=0; i < m_num_active; ++i)
			beta[m_active_set[i]] += gamma * wA[i];

		// early stopping on max l1-norm
		if (m_max_l1_norm > 0)
		{
			float64_t l1 = SGVector<float64_t>::onenorm(&beta[0], n_fea);
			if (l1 > m_max_l1_norm)
			{
				// stopping with interpolated beta
				stop_cond = true;
				lasso_cond = false;
				float64_t l1_prev = SGVector<float64_t>::onenorm(&m_beta_path[nloop][0], n_fea);
				float64_t s = (m_max_l1_norm-l1_prev)/(l1-l1_prev);

				// beta = beta_prev + s*(beta-beta_prev)
				//      = (1-s)*beta_prev + s*beta
				cblas_dscal(n_fea, s, &beta[0], 1);
				cblas_daxpy(n_fea, 1-s, &m_beta_path[nloop][0], 1, &beta[0], 1);
			}
		}

		// if lasso cond, drop the variable from active set
		if (lasso_cond)
		{
			beta[i_change] = 0; // ensure it be zero

			// update Cholesky factorization
			R=cholesky_delete(R, i_kick);
			deactivate_variable(i_kick);
		}

		nloop++;
		m_beta_path.push_back(beta);
		if (size_t(m_num_active) >= m_beta_idx.size())
			m_beta_idx.push_back(nloop);
		else
			m_beta_idx[m_num_active] = nloop;

		// early stopping with max number of non-zero variables
		if (m_max_nonz > 0 && m_num_active >= m_max_nonz)
			stop_cond = true;

	} // main loop

	// assign default estimator
	w.vlen = n_fea;
	switch_w(m_beta_idx.size()-1);

	return true;
}

SGMatrix<float64_t> CLeastAngleRegression::cholesky_insert(
		SGMatrix<float64_t> X, SGMatrix<float64_t> R, int32_t i_max_corr)
{
	// diag_k = X[:,i_max_corr]' * X[:,i_max_corr]
	float64_t diag_k = cblas_ddot(X.num_rows, X.get_column_vector(i_max_corr), 1,
		X.get_column_vector(i_max_corr), 1);

	if (m_num_active == 0)
	{ // R isn't allocated yet
		SGMatrix<float64_t> nR(1,1);
		nR(0,0) = CMath::sqrt(diag_k);
		return nR;
	}
	else
	{

		// col_k is the k-th column of (X'X)
		vector<float64_t> col_k(m_num_active);
		for (int32_t i=0; i < m_num_active; ++i)
		{
			// col_k[i] = X[:,i_max_corr]' * X[:,m_active_set[i]]
			col_k[i] = cblas_ddot(X.num_rows, X.get_column_vector(i_max_corr), 1,
				X.get_column_vector(m_active_set[i]), 1);
		}

		// R' * R_k = (X' * X)_k = col_k, solving to get R_k
		vector<float64_t> R_k(col_k);
		cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, m_num_active, 1,
			1, R.matrix, m_num_active, &R_k[0], m_num_active);

		float64_t R_kk = CMath::sqrt(diag_k -
			cblas_ddot(m_num_active, &R_k[0], 1, &R_k[0], 1));

		// new_R = [R R_k; zeros(...) R_kk]
		SGMatrix<float64_t> nR(m_num_active+1, m_num_active+1);
		for (int32_t i=0; i < m_num_active; ++i)
			for (int32_t j=0; j < m_num_active; ++j)
				nR(i,j) = R(i,j);
		for (int32_t i=0; i < m_num_active; ++i)
			nR(i, m_num_active) = R_k[i];
		for (int32_t i=0; i < m_num_active; ++i)
			nR(m_num_active, i) = 0;
		nR(m_num_active, m_num_active) = R_kk;

		return nR;
	}

}

SGMatrix<float64_t> CLeastAngleRegression::cholesky_delete(SGMatrix<float64_t> R, int32_t i_kick)
{
	if (i_kick != m_num_active-1)
	{
		// remove i_kick-th column
		for (int32_t j=i_kick; j < m_num_active-1; ++j)
			for (int32_t i=0; i < m_num_active; ++i)
				R(i,j) = R(i,j+1);

		SGMatrix<float64_t> G(2,2);
		for (int32_t i=i_kick; i < m_num_active-1; ++i)
		{
			plane_rot(R(i,i),R(i+1,i), R(i,i), R(i+1,i), G);
			if (i < m_num_active-2)
			{
				for (int32_t k=i+1; k < m_num_active-1; ++k)
				{
					// R[i:i+1, k] = G*R[i:i+1, k]
					float64_t Rik = R(i,k), Ri1k = R(i+1,k);
					R(i,k) = G(0,0)*Rik + G(0,1)*Ri1k;
					R(i+1,k) = G(1,0)*Rik+G(1,1)*Ri1k;
				}
			}
		}
	}

	SGMatrix<float64_t> nR(m_num_active-1, m_num_active-1);
	for (int32_t i=0; i < m_num_active-1; ++i)
		for (int32_t j=0; j < m_num_active-1; ++j)
			nR(i,j) = R(i,j);

	return nR;
}

#endif // HAVE_LAPACK
