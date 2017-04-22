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

CLeastAngleRegression::CLeastAngleRegression(bool lasso) :
	CLinearMachine(), m_lasso(lasso),
	m_max_nonz(0), m_max_l1_norm(0)
{
	set_epsilon(CMath::MACHINE_EPSILON);
	SG_ADD(&m_epsilon, "epsilon", "Epsilon for early stopping", MS_AVAILABLE);
	SG_ADD(&m_max_nonz, "max_nonz", "Max number of non-zero variables", MS_AVAILABLE);
	SG_ADD(&m_max_l1_norm, "max_l1_norm", "Max l1-norm of estimator", MS_AVAILABLE);
}

CLeastAngleRegression::~CLeastAngleRegression()
{
	
}

template <typename ST>
void CLeastAngleRegression::find_max_abs(const std::vector<ST> &vec, const std::vector<bool> &ignore_mask,
	int32_t &imax, ST& vmax)
{
	imax = -1;
	vmax = -1;
	for (size_t i=0; i < vec.size(); ++i)
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

template <typename ST>
void CLeastAngleRegression::plane_rot(ST x0, ST x1,
	ST &y0, ST &y1, SGMatrix<ST> &G)
{
	memset(G.matrix, 0, G.num_rows * G.num_cols * sizeof(ST));

	if (x1 == 0)
	{
		G(0, 0) = G(1, 1) = 1;
		y0 = x0;
		y1 = x1;
	}
	else
	{
		ST r = CMath::sqrt(x0*x0+x1*x1) ;
		ST sx0 = x0 / r;
		ST sx1 = x1 / r;

		G(0,0) = sx0;
		G(1,0) = -sx1;
		G(0,1) = sx1;
		G(1,1) = sx0;

		y0 = r;
		y1 = 0;
	}
}

bool CLeastAngleRegression::train_machine(CFeatures* data)
{
	REQUIRE(m_labels->get_label_type() == LT_REGRESSION, "Provided labels (%s) are of type (%d) - they should be regression labels (%d) instead.\n"
		, m_labels->get_name(), m_labels->get_label_type(), LT_REGRESSION, m_labels->get_label_type())

	if (!data)
	{
		REQUIRE(features, "No features provided.\n")
		REQUIRE(features->get_feature_class() == C_DENSE,
			"Feature-class (%d) must be of type C_DENSE (%d)\n", features->get_feature_class(), C_DENSE)
			
		data = features;
	}
	else
		REQUIRE(data->get_feature_class() == C_DENSE,
			"Feature-class must be of type C_DENSE (%d)\n", data->get_feature_class(), C_DENSE)

	REQUIRE(data->get_num_vectors() == m_labels->get_num_labels(), "Number of training vectors (%d) does not match number of labels (%d)\n"
		, data->get_num_vectors(), m_labels->get_num_labels())

	//check for type of CFeatures, then call the appropriate template method
	if(data->get_feature_type() == F_DREAL)
		return CLeastAngleRegression::train_machine_templated((CDenseFeatures<float64_t> *) data);
	else if(data->get_feature_type() == F_SHORTREAL)
		return CLeastAngleRegression::train_machine_templated((CDenseFeatures<float32_t> *) data);
	else if(data->get_feature_type() == F_LONGREAL)
		return CLeastAngleRegression::train_machine_templated((CDenseFeatures<floatmax_t> *) data);
	else
		SG_ERROR("Feature-type (%d) must be of type F_SHORTREAL (%d), F_DREAL (%d) or F_LONGREAL (%d).\n", 
			data->get_feature_type(), F_SHORTREAL, F_DREAL, F_LONGREAL)

	return false;
}

template <typename ST>
bool CLeastAngleRegression::train_machine_templated(CDenseFeatures<ST> * data)
{
	std::vector<std::vector<ST>> m_beta_path_t;		

	int32_t n_fea = data->get_num_features();
	int32_t n_vec = data->get_num_vectors();

	bool lasso_cond = false;
	bool stop_cond = false;

	// init facilities
	m_beta_idx.clear();
	m_beta_path_t.clear();
	m_num_active = 0;
	m_active_set.clear();
	m_is_active.resize(n_fea);
	fill(m_is_active.begin(), m_is_active.end(), false);

	SGVector<ST> y = ((CRegressionLabels*) m_labels)->template get_labels_t<ST>();
	typename SGVector<ST>::EigenVectorXtMap map_y(y.vector, y.size());

	// transpose(X) is more convenient to work with since we care
	// about features here. After transpose, each row will be a data
	// point while each column corresponds to a feature
	SGMatrix<ST> X (n_vec, n_fea);
	typename SGMatrix<ST>::EigenMatrixXtMap map_Xr = data->get_feature_matrix();
	typename SGMatrix<ST>::EigenMatrixXtMap map_X(X.matrix, n_vec, n_fea);
	map_X = map_Xr.transpose();

	SGMatrix<ST> X_active(n_vec, n_fea);

	// beta is the estimator
	vector<ST> beta(n_fea);

	vector<ST> Xy(n_fea);
	typename SGVector<ST>::EigenVectorXtMap map_Xy(&Xy[0], n_fea);
	// Xy = X' * y
	map_Xy=map_Xr*map_y;

	// mu is the prediction
	vector<ST> mu(n_vec);

	// correlation
	vector<ST> corr(n_fea);
	// sign of correlation
	vector<ST> corr_sign(n_fea);

	// Cholesky factorization R'R = X'X, R is upper triangular
	SGMatrix<ST> R;

	ST max_corr = 1;
	int32_t i_max_corr = 1;

	// first entry: all coefficients are zero
	m_beta_path_t.push_back(beta);
	m_beta_idx.push_back(0);

	//maximum allowed active variables at a time
	int32_t max_active_allowed = CMath::min(n_vec-1, n_fea);

	//========================================
	// main loop
	//========================================
	int32_t nloop=0;
	while (m_num_active < max_active_allowed && max_corr/n_vec > get_epsilon() && !stop_cond)
	{
		// corr = X' * (y-mu) = - X'*mu + Xy
		typename SGVector<ST>::EigenVectorXtMap map_corr(&corr[0], n_fea);
		typename SGVector<ST>::EigenVectorXtMap map_mu(&mu[0], n_vec);
		
		map_corr = map_Xy - (map_Xr*map_mu);
		
		// corr_sign = sign(corr)
		for (size_t i=0; i < corr.size(); ++i)
			corr_sign[i] = CMath::sign(corr[i]);

		// find max absolute correlation in inactive set
		find_max_abs(corr, m_is_active, i_max_corr, max_corr);

		if (!lasso_cond)
		{
			// update Cholesky factorization matrix
			if (m_num_active == 0)
			{ 
				// R isn't allocated yet
				R=SGMatrix<ST>(1,1);
				ST diag_k = map_X.col(i_max_corr).dot(map_X.col(i_max_corr));
				R(0,0) = CMath::sqrt( diag_k);
			}
			else
				R=cholesky_insert(X, X_active, R, i_max_corr, m_num_active);
			activate_variable(i_max_corr);
		}

		// Active variables
		typename SGMatrix<ST>::EigenMatrixXtMap map_Xa(X_active.matrix, n_vec, m_num_active);
		if (!lasso_cond)
			map_Xa.col(m_num_active-1)=map_X.col(i_max_corr);
		
		SGVector<ST> corr_sign_a(m_num_active);
		for (index_t i=0; i < m_num_active; ++i)
			corr_sign_a[i] = corr_sign[m_active_set[i]];

		typename SGVector<ST>::EigenVectorXtMap map_corr_sign_a(corr_sign_a.vector, corr_sign_a.size());
		typename SGMatrix<ST>::EigenMatrixXtMap map_R(R.matrix, R.num_rows, R.num_cols);
		typename SGVector<ST>::EigenVectorXt solve = map_R.transpose().template triangularView<Lower>().template solve<OnTheLeft>(map_corr_sign_a);

		typename SGVector<ST>::EigenVectorXt GA1 = map_R.template triangularView<Upper>().template solve<OnTheLeft>(solve);

		// AA = 1/sqrt(GA1' * corr_sign_a)
		ST AA = GA1.dot(map_corr_sign_a);
		AA = 1/CMath::sqrt( AA);
		
		typename SGVector<ST>::EigenVectorXt wA = AA*GA1;

		// equiangular direction (unit vector)
		vector<ST> u(n_vec);
		typename SGVector<ST>::EigenVectorXtMap map_u(&u[0], n_vec);
		
		map_u = map_Xa*wA;

		ST gamma = max_corr / AA;
		if (m_num_active < n_fea)
		{
			#pragma omp parallel for shared(gamma)
			for (index_t i=0; i < n_fea; ++i)
			{
				if (m_is_active[i])
					continue;

				// correlation between X[:,i] and u
				ST dir_corr = map_u.dot(map_X.col(i));

				ST tmp1 = (max_corr-corr[i])/(AA-dir_corr);
				ST tmp2 = (max_corr+corr[i])/(AA+dir_corr);
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
			ST lasso_bound = numeric_limits<ST>::max();

			for (index_t i=0; i < m_num_active; ++i)
			{
				ST tmp = -beta[m_active_set[i]] / wA(i);
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
		for (index_t i=0; i < m_num_active; ++i)
			beta[m_active_set[i]] += gamma * wA(i);

		// early stopping on max l1-norm
		if (get_max_l1_norm() > 0)
		{
			ST l1 = SGVector<ST>::onenorm(&beta[0], n_fea);
			if (l1 > get_max_l1_norm())
			{
				// stopping with interpolated beta
				stop_cond = true;
				lasso_cond = false;
				ST l1_prev = (ST) SGVector<ST>::onenorm(&m_beta_path_t[nloop][0], n_fea);
				ST s = (get_max_l1_norm()-l1_prev)/(l1-l1_prev);

				typename SGVector<ST>::EigenVectorXtMap map_beta(&beta[0], n_fea);
				typename SGVector<ST>::EigenVectorXtMap map_beta_prev(&m_beta_path_t[nloop][0], n_fea);
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
		m_beta_path_t.push_back(beta);
		if (size_t(m_num_active) >= get_path_size())
			m_beta_idx.push_back(nloop);
		else
			m_beta_idx[m_num_active] = nloop;

		// early stopping with max number of non-zero variables
		if (get_max_non_zero() > 0 && m_num_active >= get_max_non_zero())
			stop_cond = true;
		SG_DEBUG("Added : %d , Dropped %d, Active set size %d max_corr %.17f \n", i_max_corr, i_kick, m_num_active, max_corr);
	}

	//copy m_beta_path_t (of type ST) into m_beta_path
	for(size_t i = 0; i < m_beta_path_t.size(); ++i)
	{
		std::vector<float64_t> va;
		for(size_t p = 0; p < m_beta_path_t[i].size(); ++p){
			va.push_back((float64_t) m_beta_path_t[i][p]);			
		}
		m_beta_path.push_back(va);
	}

	// assign default estimator
	set_w(SGVector<float64_t>(n_fea));
	switch_w(get_path_size()-1);
	
	return true;
}

template <typename ST>
SGMatrix<ST> CLeastAngleRegression::cholesky_insert(const SGMatrix<ST>& X, 
		const SGMatrix<ST>& X_active, SGMatrix<ST>& R, int32_t i_max_corr, int32_t num_active)
{
	typename SGMatrix<ST>::EigenMatrixXtMap map_X(X.matrix, X.num_rows, X.num_cols);
	typename SGMatrix<ST>::EigenMatrixXtMap map_X_active(X_active.matrix, X.num_rows, num_active);
	ST diag_k = map_X.col(i_max_corr).dot(map_X.col(i_max_corr));
	
	// col_k is the k-th column of (X'X)
	typename SGVector<ST>::EigenVectorXtMap map_i_max(X.get_column_vector(i_max_corr), X.num_rows);
	typename SGVector<ST>::EigenVectorXt R_k = map_X_active.transpose()*map_i_max;
	typename SGMatrix<ST>::EigenMatrixXtMap map_R(R.matrix, R.num_rows, R.num_cols);

	// R' * R_k = (X' * X)_k = col_k, solving to get R_k
	map_R.transpose().template triangularView<Lower>().template solveInPlace<OnTheLeft>(R_k);
	ST R_kk = CMath::sqrt(diag_k - R_k.dot(R_k));

	SGMatrix<ST> R_new(num_active+1, num_active+1);
	typename SGMatrix<ST>::EigenMatrixXtMap map_R_new(R_new.matrix, R_new.num_rows, R_new.num_cols);

	map_R_new.block(0, 0, num_active, num_active) = map_R;
	sg_memcpy(R_new.matrix+num_active*(num_active+1), R_k.data(), sizeof(ST)*(num_active));
	map_R_new.row(num_active).setZero();
	map_R_new(num_active, num_active) = R_kk;
	return R_new;
}

template <typename ST>
SGMatrix<ST> CLeastAngleRegression::cholesky_delete(SGMatrix<ST>& R, int32_t i_kick)
{
	if (i_kick != m_num_active-1)
	{
		// remove i_kick-th column
		for (index_t j=i_kick; j < m_num_active-1; ++j)
			for (index_t i=0; i < m_num_active; ++i)
				R(i,j) = R(i,j+1);

		SGMatrix<ST> G(2,2);
		for (index_t i=i_kick; i < m_num_active-1; ++i)
		{
			plane_rot(R(i,i),R(i+1,i), R(i,i), R(i+1,i), G);
			if (i < m_num_active-2)
			{
				for (index_t k=i+1; k < m_num_active-1; ++k)
				{
					// R[i:i+1, k] = G*R[i:i+1, k]
					ST Rik = R(i,k), Ri1k = R(i+1,k);
					R(i,k) = G(0,0)*Rik + G(0,1)*Ri1k;
					R(i+1,k) = G(1,0)*Rik+G(1,1)*Ri1k;
				}
			}
		}
	}

	SGMatrix<ST> nR(m_num_active-1, m_num_active-1);
	for (index_t i=0; i < m_num_active-1; ++i)
		for (index_t j=0; j < m_num_active-1; ++j)
			nR(i,j) = R(i,j);

	return nR;
}

template bool CLeastAngleRegression::train_machine_templated<float64_t>(CDenseFeatures<float64_t> * data);
template bool CLeastAngleRegression::train_machine_templated<float32_t>(CDenseFeatures<float32_t> * data);
template SGMatrix<float32_t> CLeastAngleRegression::cholesky_insert(const SGMatrix<float32_t>& X, const SGMatrix<float32_t>& X_active, SGMatrix<float32_t>& R, int32_t i_max_corr, int32_t num_active);
template SGMatrix<float64_t> CLeastAngleRegression::cholesky_insert(const SGMatrix<float64_t>& X, const SGMatrix<float64_t>& X_active, SGMatrix<float64_t>& R, int32_t i_max_corr, int32_t num_active);
