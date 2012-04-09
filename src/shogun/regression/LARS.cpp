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
#include <algorithm>

#include <shogun/regression/LARS.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;
using namespace std;

static vector<float64_t> make_vector(int32_t size, float64_t val)
{
	vector<float64_t> result(size);
	fill(result.begin(), result.end(), val);
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

	// TODO: centralize y

	CSimpleFeatures<float64_t>* feats=(CSimpleFeatures<float64_t>*) data;
	int32_t n_fea = feats->get_num_features();
	int32_t n_smp = feats->get_num_vectors();

	bool lasso_cond = false;

	int32_t n_active = 0;
	vector<int32_t> active_set;
	vector<bool> is_active(n_fea);
	fill(is_active.begin(), is_active.end(), false);

	SGVector<float64_t> y = m_labels->get_labels();
	SGMatrix<float64_t> X = data->get_feature_matrix();

	// X * y
	vector<float64_t> Xy(n_fea);
	fill(Xy.begin(), Xy.end(), 0);
	// Xy = X * y
	cblas_dgemv(CblasColMajor, CblasNoTrans, n_fea, n_smp, 1, X.matrix, n_fea, 
		y.vector, 1, 0, &Xy[0], 1);

	// \mu is the prediction
	vector<float64_t> mu = make_vector(n_smp, 0);

	// correlation
	vector<float64_t> corr(Xy); // init to X*y
	// sign of correlation
	vector<float64_t> corr_sign(n_fea);
	for (int32_t i=0; i < corr.size(); ++i)
		corr_sign[i] = CMath::sign(corr[i]);

	// Gram matrix
	SGMatrix<float64_t> Gram(n_smp, n_smp);
	Gram.zero();
	for (int32_t i=0; i < n_smp; ++i)
	{
		// Gram += X[:,i] * X[:,i]'
		cblas_dger(CblasColMajor, n_smp, n_smp, 1, X.get_column_vector(i), 1, 
			X.get_column_vector(i), 1, Gram.matrix, n_smp);
	}

	while (n_active < n_fea && max_corr > CMath::MACHINE_EPSILON)
	{
		float64_t max_corr = -1;
		int32_t i_max_corr = -1;
		for (int32_t i=0; i < corr.size(); ++i)
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
			active_set.push_back(i_max_corr);
			is_active[i_max_corr] = true;
			n_active++;

		SGMatrix<float64_t> Gram_sub(n_active, n_active);
		for (int32_t i=0; i < n_active; ++i)
		{
			for (int32_t j=i; j < n_active; ++j)
			{
				Gram_sub(i,j) = corr_sign[i]*corr_sign[j]*
					Gram(active_set[i], active_set[j]);
				Gram_sub(j,i) = Gram_sub(i,j);
			}
		}
		vector<float64_t> one = make_vector(n_active, 1);
		vector<float64_t> LA(n_active);
		// LA = (one' * inv(Gram_sub) * one)^(-1/2)
		// TODO: compute inv(Gram_sub)
		for (int i=0; i < n_active; ++i)
			LA[i] = 1/CMath::sqrt(LA[i]);


	}

	y.destroy_vector();
	X.destroy_matrix();
}
