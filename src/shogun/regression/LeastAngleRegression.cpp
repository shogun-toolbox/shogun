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

#include <vector>
#include <limits>
#include <algorithm>

#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/regression/LeastAngleRegression.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/eigen3.h>

using namespace Eigen;
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
	m_epsilon = CMath::MACHINE_EPSILON;
	SG_ADD(&m_epsilon, "epsilon", "Epsilon for early stopping", MS_AVAILABLE);
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
	Map<VectorXd> map_y(y.vector, y.size());
	SGMatrix<float64_t> Xorig = feats->get_feature_matrix();

	// transpose(X) is more convenient to work with since we care
	// about features here. After transpose, each row will be a data
	// point while each column corresponds to a feature
	SGMatrix<float64_t> X (n_vec, n_fea);
	Map<MatrixXd> map_Xr(Xorig.matrix, n_fea, n_vec);
	Map<MatrixXd> map_X(X.matrix, n_vec, n_fea);
	map_X = map_Xr.transpose();
	
	SGMatrix<float64_t> X_active(n_vec, n_fea);

	// beta is the estimator
	vector<float64_t> beta = make_vector(n_fea, 0);

	vector<float64_t> Xy = make_vector(n_fea, 0);
	Map<VectorXd> map_Xy(&Xy[0], n_fea);
	// Xy = X' * y
	map_Xy=map_Xr*map_y;

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
	while (m_num_active < max_active_allowed && max_corr/n_vec > m_epsilon && !stop_cond)
	{
		// corr = X' * (y-mu) = - X'*mu + Xy
		Map<VectorXd> map_corr(&corr[0], n_fea);
		Map<VectorXd> map_mu(&mu[0], n_vec);
		
		map_corr = map_Xy - (map_Xr*map_mu);
		
		// corr_sign = sign(corr)
		for (uint32_t i=0; i < corr.size(); ++i)
			corr_sign[i] = CMath::sign(corr[i]);

		// find max absolute correlation in inactive set
		find_max_abs(corr, m_is_active, i_max_corr, max_corr);

		if (!lasso_cond)
		{
			// update Cholesky factorization matrix
			if (m_num_active == 0)
			{ 
				// R isn't allocated yet
				R=SGMatrix<float64_t>(1,1);
				float64_t diag_k = map_X.col(i_max_corr).dot(map_X.col(i_max_corr));
				R(0,0) = CMath::sqrt(diag_k);
			}
			else
				R=cholesky_insert(X, X_active, R, i_max_corr, m_num_active);
			activate_variable(i_max_corr);
		}

		// Active variables
		Map<MatrixXd> map_Xa(X_active.matrix, n_vec, m_num_active);
		if (!lasso_cond)
			map_Xa.col(m_num_active-1)=map_X.col(i_max_corr);
		
		SGVector<float64_t> corr_sign_a(m_num_active);
		for (int32_t i=0; i < m_num_active; ++i)
			corr_sign_a[i] = corr_sign[m_active_set[i]];

		Map<VectorXd> map_corr_sign_a(corr_sign_a.vector, corr_sign_a.size());
		Map<MatrixXd> map_R(R.matrix, R.num_rows, R.num_cols);
		VectorXd solve = map_R.transpose().triangularView<Lower>().solve<OnTheLeft>(map_corr_sign_a);
		VectorXd GA1 = map_R.triangularView<Upper>().solve<OnTheLeft>(solve);

		// AA = 1/sqrt(GA1' * corr_sign_a)
		float64_t AA = GA1.dot(map_corr_sign_a);
		AA = 1/CMath::sqrt(AA);

		VectorXd wA = AA*GA1;

		// equiangular direction (unit vector)
		vector<float64_t> u = make_vector(n_vec, 0);
		Map<VectorXd> map_u(&u[0], n_vec);
		
		map_u = map_Xa*wA;

		float64_t gamma = max_corr / AA;
		if (m_num_active < n_fea)
		{
			#pragma omp parallel for shared(gamma)
			for (int32_t i=0; i < n_fea; ++i)
			{
				if (m_is_active[i])
					continue;

				// correlation between X[:,i] and u
				float64_t dir_corr = map_u.dot(map_X.col(i));

				float64_t tmp1 = (max_corr-corr[i])/(AA-dir_corr);
				float64_t tmp2 = (max_corr+corr[i])/(AA+dir_corr);
				#pragma omp critical
				{
				if (tmp1 > CMath::MACHINE_EPSILON && tmp1 < gamma)
					gamma = tmp1;
				if (tmp2 > CMath::MACHINE_EPSILON && tmp2 < gamma)
					gamma = tmp2;
				}
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
				float64_t tmp = -beta[m_active_set[i]] / wA(i);
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
		map_mu += gamma*map_u;

		// update estimator
		for (int32_t i=0; i < m_num_active; ++i)
			beta[m_active_set[i]] += gamma * wA(i);
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

				Map<VectorXd> map_beta(&beta[0], n_fea);
				Map<VectorXd> map_beta_prev(&m_beta_path[nloop][0], n_fea);
				map_beta = (1-s)*map_beta_prev + s*map_beta;
			}
		}

		// if lasso cond, drop the variable from active set
		if (lasso_cond)
		{
			beta[i_change] = 0; 
			R=cholesky_delete(R, i_kick);
			deactivate_variable(i_kick);

			// Remove column from active set
			int32_t numRows = map_Xa.rows();
			int32_t numCols = map_Xa.cols()-1;
			if( i_kick < numCols )
				map_Xa.block(0, i_kick, numRows, numCols-i_kick) = 
					map_Xa.block(0, i_kick+1, numRows, numCols-i_kick).eval();			
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
		SG_DEBUG("Added : %d , Dropped %d, Active set size %d max_corr %.17f \n", i_max_corr, i_kick, m_num_active, max_corr);

	}

	// assign default estimator
	w.vlen = n_fea;
	switch_w(m_beta_idx.size()-1);

	return true;
}

SGMatrix<float64_t> CLeastAngleRegression::cholesky_insert(const SGMatrix<float64_t>& X, 
		const SGMatrix<float64_t>& X_active, SGMatrix<float64_t>& R, int32_t i_max_corr, int32_t num_active)
{
	Map<MatrixXd> map_X(X.matrix, X.num_rows, X.num_cols);
	Map<MatrixXd> map_X_active(X_active.matrix, X.num_rows, num_active);
	float64_t diag_k = map_X.col(i_max_corr).dot(map_X.col(i_max_corr));
	
	// col_k is the k-th column of (X'X)
	Map<VectorXd> map_i_max(X.get_column_vector(i_max_corr), X.num_rows);
	VectorXd R_k = map_X_active.transpose()*map_i_max;
	Map<MatrixXd> map_R(R.matrix, R.num_rows, R.num_cols);

	// R' * R_k = (X' * X)_k = col_k, solving to get R_k
	map_R.transpose().triangularView<Lower>().solveInPlace<OnTheLeft>(R_k);
	float64_t R_kk = CMath::sqrt(diag_k - R_k.dot(R_k));

	SGMatrix<float64_t> R_new(num_active+1, num_active+1);
	Map<MatrixXd> map_R_new(R_new.matrix, R_new.num_rows, R_new.num_cols);

	map_R_new.block(0, 0, num_active, num_active) = map_R;
	memcpy(R_new.matrix+num_active*(num_active+1), R_k.data(), sizeof(float64_t)*(num_active));
	map_R_new.row(num_active).setZero();
	map_R_new(num_active, num_active) = R_kk;
	return R_new;
}

SGMatrix<float64_t> CLeastAngleRegression::cholesky_delete(SGMatrix<float64_t>& R, int32_t i_kick)
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

