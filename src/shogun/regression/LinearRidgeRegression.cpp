/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Soeren Sonnenburg
 */

#include <lib/config.h>

#ifdef HAVE_LAPACK
#include <regression/LinearRidgeRegression.h>
#include <mathematics/lapack.h>
#include <mathematics/Math.h>
#include <labels/RegressionLabels.h>

using namespace shogun;

CLinearRidgeRegression::CLinearRidgeRegression()
: CLinearMachine()
{
	init();
}

CLinearRidgeRegression::CLinearRidgeRegression(float64_t tau, CDenseFeatures<float64_t>* data, CLabels* lab)
: CLinearMachine()
{
	init();

	m_tau=tau;
	set_labels(lab);
	set_features(data);
}

void CLinearRidgeRegression::init()
{
	m_tau=1e-6;

	SG_ADD(&m_tau, "tau", "Regularization parameter", MS_AVAILABLE);
}

bool CLinearRidgeRegression::train_machine(CFeatures* data)
{
	if (!m_labels)
		SG_ERROR("No labels set\n")

	if (!data)
		data=features;

	if (!data)
		SG_ERROR("No features set\n")

	if (m_labels->get_num_labels() != data->get_num_vectors())
		SG_ERROR("Number of training vectors does not match number of labels\n")

	if (data->get_feature_class() != C_DENSE)
		SG_ERROR("Expected Dense Features\n")

	if (data->get_feature_type() != F_DREAL)
		SG_ERROR("Expected Real Features\n")

	CDenseFeatures<float64_t>* feats=(CDenseFeatures<float64_t>*) data;
	int32_t num_feat=feats->get_num_features();
	int32_t num_vec=feats->get_num_vectors();

	// Get kernel matrix
	SGMatrix<float64_t> kernel_matrix(num_feat,num_feat);
	SGVector<float64_t> y(num_feat);

	// init
	kernel_matrix.zero();
	y.zero();

	for (int32_t i=0; i<num_feat; i++)
		kernel_matrix.matrix[i+i*num_feat]+=m_tau;

	for (int32_t i=0; i<num_vec; i++)
	{
		SGVector<float64_t> v = feats->get_feature_vector(i);
		ASSERT(v.vlen==num_feat)

		cblas_dger(CblasColMajor, num_feat,num_feat, 1.0, v.vector,1,
				v.vector,1, kernel_matrix.matrix, num_feat);

		cblas_daxpy(num_feat, ((CRegressionLabels*) m_labels)->get_label(i), v.vector, 1, y.vector, 1);

		feats->free_feature_vector(v, i);
	}

	clapack_dposv(CblasRowMajor,CblasUpper, num_feat, 1, kernel_matrix.matrix, num_feat,
			y.vector, num_feat);

	set_w(y);

	return true;
}

bool CLinearRidgeRegression::load(FILE* srcfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

bool CLinearRidgeRegression::save(FILE* dstfile)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}
#endif
