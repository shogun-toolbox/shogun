/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <vector>
#include <limits>
#include <algorithm>
#include <cstdio> // TODO: remove debug code

#include <shogun/lib/config.h>

#ifdef HAVE_LAPACK
#include <shogun/features/SimpleFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/regression/LARS.h>

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

bool LARS::train_machine(CFeatures* data)
{
	if (!m_labels)
		SG_ERROR("No labels set\n");

	if (!data)
		data=features;

	if (!data)
		SG_ERROR("No features set\n");

	if (m_labels->get_num_labels() != data->get_num_vectors())
		SG_ERROR("Number of training vectors does not match number of labels\n");

	if (data->get_feature_class() != C_SIMPLE)
		SG_ERROR("Expected Simple Features\n");

	if (data->get_feature_type() != F_DREAL)
		SG_ERROR("Expected Real Features\n");

	// TODO: X should be normalized (zero mean, unit length) and y should
	// be centered (zero mean)
	printf("LARS::train> prepare data...\n");

	CSimpleFeatures<float64_t>* feats=(CSimpleFeatures<float64_t>*) data;
	int32_t n_fea = feats->get_num_features();
	int32_t n_vec = feats->get_num_vectors();

	bool lasso_cond = false;

	int32_t n_active = 0;
	vector<int32_t> active_set;
	vector<bool> is_active(n_fea);
	fill(is_active.begin(), is_active.end(), false);

	SGVector<float64_t> y = m_labels->get_labels();
	SGMatrix<float64_t> Xorig = feats->get_feature_matrix();

	// transpose(X) is more convenient to work with since we care
	// about features here. After transpose, each row will be a data
	// point while each column corresponds to a feature
	SGMatrix<float64_t> X(n_vec, n_fea, true);
	for (int32_t i=0; i < n_vec; ++i)
		for (int32_t j=0; j < n_fea; ++j)
			X(i,j) = Xorig(j,i);
	Xorig.free_matrix();
	
	printf("LARS::train> init variables...\n");

	// beta is the regressor
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

	printf("LARS::train> enter main loop...\n");

	while (n_active < n_fea && max_corr > CMath::MACHINE_EPSILON)
	{
		printf("    \nLARS::loop> find correlation...\n");
		// corr = X' * (y-mu) = - X'*mu + Xy
		copy(Xy.begin(), Xy.end(), corr.begin());
		cblas_dgemv(CblasColMajor, CblasTrans, n_vec, n_fea, -1, 
			X.matrix, n_vec, &mu[0], 1, 1, &corr[0], 1);

		// corr_sign = sign(corr)
		for (uint32_t i=0; i < corr.size(); ++i)
			corr_sign[i] = CMath::sign(corr[i]);

		max_corr = -1;
		i_max_corr = -1;
		// find max absolute correlation in inactive set
		for (uint32_t i=0; i < corr.size(); ++i)
		{
			if (is_active[i])
				continue;
			if (CMath::abs(corr[i]) > max_corr)
			{
				max_corr = CMath::abs(corr[i]);
				i_max_corr = i;
			}
		}

		if (!lasso_cond)
		{
			//--------------------------------------
			// update Cholesky factorization matrix
			//--------------------------------------

			printf("    LARS::loop> update cholesky...\n");
			// diag_k = X[:,i_max_corr]' * X[:,i_max_corr]
			float64_t diag_k = cblas_ddot(n_vec, X.get_column_vector(i_max_corr), 1,
				X.get_column_vector(i_max_corr), 1);

			if (n_active == 0)
			{ // R isn't allocated yet
				R.matrix = SG_MALLOC(float64_t, 1);
				R.num_rows = 1;
				R.num_cols = 1;
				R.matrix[0] = CMath::sqrt(diag_k);
			}
			else
			{
				float64_t *new_R = SG_MALLOC(float64_t, (n_active+1)*(n_active+1));

				// col_k is the k-th column of (X'X)
				vector<float64_t> col_k(n_active);
				for (int32_t i=0; i < n_active; ++i)
				{
					// col_k[i] = X[:,i_max_corr]' * X[:,active_set[i]]
					col_k[i] = cblas_ddot(n_vec, X.get_column_vector(i_max_corr), 1,
						X.get_column_vector(active_set[i]), 1);
				}

				// R' * R_k = (X' * X)_k = col_k, solving to get R_k
				vector<float64_t> R_k(col_k);
				cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, n_active, 1, 
					1, R.matrix, n_active, &R_k[0], n_active);

				float64_t R_kk = CMath::sqrt(diag_k - 
					cblas_ddot(n_active, &R_k[0], 1, &R_k[0], 1));

				// new_R = [R R_k; zeros(...) R_kk]
				SGMatrix<float64_t> nR;
				nR.matrix = new_R;
				nR.num_rows = n_active+1;
				nR.num_cols = n_active+1;
				for (int32_t i=0; i < n_active; ++i)
					for (int32_t j=0; j < n_active; ++j)
						nR(i,j) = R(i,j);
				for (int32_t i=0; i < n_active; ++i)
					nR(i, n_active) = R_k[i];
				for (int32_t i=0; i < n_active; ++i)
					nR(n_active, i) = 0;
				nR(n_active, n_active) = R_kk;

				// update R
				SG_FREE(R.matrix);
				R.matrix = nR.matrix;
				R.num_rows = nR.num_rows;
				R.num_cols = nR.num_cols;
			}

			printf("    LARS::loop> add new variable %d...\n", i_max_corr);
			// add new variable to active set
			active_set.push_back(i_max_corr);
			is_active[i_max_corr] = true;
			n_active++;
		} // if (!lasso_cond)


		printf("    LARS::loop> compute aux values...\n");

		// GA1 = R\(R'\corr_sign)
		vector<float64_t> GA1(corr_sign);
		cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit, 
			n_active, 1, 1, R.matrix, n_active, &GA1[0], n_active);
		cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
			n_active, 1, 1, R.matrix, n_active, &GA1[0], n_active);

		// AA = 1/sqrt(GA1' * corr_sign)
		float64_t AA = cblas_ddot(n_active, &GA1[0], 1, &corr_sign[0], 1);
		AA = 1/CMath::sqrt(AA);

		// wA = AA*GA1
		vector<float64_t> wA(GA1);
		for (int32_t i=0; i < n_active; ++i)
			wA[i] *= AA;

		// equiangular direction (unit vector)
		vector<float64_t> u = make_vector(n_vec, 0);
		// u = X[:,active_set] * wA
		for (int32_t i=0; i < n_active; ++i)
		{
			// u += wA[i] * X[:,active_set[i]]
			cblas_daxpy(n_vec, wA[i], 
				X.get_column_vector(active_set[i]), 1, &u[0], 1);
		}

		printf("    LARS::loop> find suitable step size...\n");

		// step size
		float64_t gamma = max_corr / AA; 
		if (n_active < n_fea)
		{
			for (int32_t i=0; i < n_fea; ++i)
			{
				if (is_active[i])
					continue;

				// correlation between X[:,i] and u
				float64_t dir_corr = cblas_ddot(n_vec, 
					X.get_column_vector(i), 1, &u[0], 1);

				float64_t tmp1 = (max_corr-corr[i])/(AA-dir_corr);
				float64_t tmp2 = (max_corr+corr[i])/(AA+dir_corr);
				if (tmp1 > 0 && tmp1 < gamma)
					gamma = tmp1;
				if (tmp2 > 0 && tmp2 < gamma)
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

			for (int32_t i=0; i < n_active; ++i)
			{
				float64_t tmp = -beta[active_set[i]] / wA[i];
				if (tmp > 0 && tmp < lasso_bound)
				{
					lasso_bound = tmp;
					i_kick = i;
				}
			}

			if (lasso_bound < gamma)
			{
				gamma = lasso_bound;
				lasso_cond = true;
				i_change = active_set[i_kick];
			}
		}

		printf("    LARS::loop> update estimations...\n");

		// update prediction: mu = mu + gamma * u
		cblas_daxpy(n_vec, gamma, &u[0], 1, &mu[0], 1);

		// update regressor
		for (int32_t i=0; i < n_active; ++i)
			beta[active_set[i]] += gamma * wA[i];

		// TODO: record beta along the path

		// if lasso cond, drop the variable from active set
		if (lasso_cond)
		{
			printf("    LARS::loop> lasso cond...\n");

			// -------------------------------------------
			// update Cholesky factorization
			// -------------------------------------------
			SGMatrix<float64_t> nR(n_active-1, n_active-1);
			if (i_kick != n_active-1)
			{
				// remove i_kick-th column
				for (int32_t j=i_kick; j < n_active-1; ++j)
					for (int32_t i=0; i < n_active; ++i)
						R(i,j) = R(i,j+1);

				SGMatrix<float64_t> G(2,2);
				for (int32_t i=i_kick; i < n_active-1; ++i)
				{
					plane_rot(R(i,i),R(i+1,i), 
						R(i,i), R(i+1,i), G);
					if (i < n_active-2)
					{
						for (int32_t k=i+1; k < n_active-1; ++k)
						{
							// R[i:i+1, k] = G*R[i:i+1, k]
							R(i,k) = G(0,0)*R(i,k) + G(0,1)*R(i+1,k);
							R(i+1,k) = G(1,0)*R(i,k)+G(1,1)*R(i+1,k);
						}
					}
				}

				G.destroy_matrix();

			}
			for (int32_t i=0; i < n_active-1; ++i)
				for (int32_t j=0; j < n_active-1; ++j)
					nR(i,j) = R(i,j);

			SG_FREE(R.matrix);
			R.matrix = nR.matrix;
			R.num_cols = nR.num_cols;
			R.num_rows = nR.num_rows;

			// remove this variable
			active_set.erase(active_set.begin() + i_kick);
			is_active[i_change] = false;
			n_active--;
		}
	}

	printf("LARS::train> exit main loop...\n");

	y.free_vector();
	X.free_matrix();
	// TODO: destroy R

	return true;
}

bool LARS::load(FILE* srcfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

bool LARS::save(FILE* dstfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}
#endif // HAVE_LAPACK
