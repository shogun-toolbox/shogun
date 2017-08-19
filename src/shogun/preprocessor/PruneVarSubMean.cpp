/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/preprocessor/PruneVarSubMean.h>
#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

using namespace shogun;

CPruneVarSubMean::CPruneVarSubMean(bool divide)
: CDensePreprocessor<float64_t>()
{
	init();
	register_parameters();
	m_divide_by_std = divide;
}

CPruneVarSubMean::~CPruneVarSubMean()
{
	cleanup();
}

/// initialize preprocessor from features
bool CPruneVarSubMean::init(CFeatures* features)
{
	if (!m_initialized)
	{
		ASSERT(features->get_feature_class()==C_DENSE)
		ASSERT(features->get_feature_type()==F_DREAL)

		CDenseFeatures<float64_t>* simple_features=(CDenseFeatures<float64_t>*) features;
		int32_t num_features = simple_features->get_num_features();

		auto mean = simple_features->get_mean();
		auto std = simple_features->get_std();
		std::vector<index_t> idx;
		index_t num_ok = 0;

		for (index_t j=0; j<num_features; ++j)
		{
			if (std[j]>=1e-7)
			{
				idx.push_back(j);
				++num_ok;
			}
		}

		SG_INFO("Reducing number of features from %i to %i\n", num_features, num_ok)

		m_idx = SGVector<int32_t>(num_ok);
		m_mean = SGVector<float64_t>(num_ok);
		m_std = SGVector<float64_t>(num_ok);

		for (index_t j=0; j<num_ok; ++j)
		{
			m_idx[j]=idx[j];
			m_mean[j]=mean[idx[j]];
			m_std[j]=std[idx[j]];
		}
		m_num_idx = num_ok;
		m_initialized = true;
		return true;
	}
	else
		return false;
}

/// clean up allocated memory
void CPruneVarSubMean::cleanup()
{
	m_idx=SGVector<int32_t>();
	m_mean=SGVector<float64_t>();
	m_std=SGVector<float64_t>();
	m_initialized = false;
}

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
SGMatrix<float64_t> CPruneVarSubMean::apply_to_feature_matrix(CFeatures* features)
{
	ASSERT(m_initialized)

	auto src=((CDenseFeatures<float64_t>*) features)->get_feature_matrix();
	auto num_vectors = src.num_cols;
	SGMatrix<float64_t> dst(m_num_idx, num_vectors);

	SG_INFO("Preprocessing feature matrix\n")
	for (index_t vec=0; vec<num_vectors; ++vec)
	{
		if (m_divide_by_std)
		{
			for (int32_t feat=0; feat<m_num_idx; feat++)
				dst[feat]=(src[m_idx[feat]]-m_mean[feat])/m_std[feat];
		}
		else
		{
			for (int32_t feat=0; feat<m_num_idx; feat++)
				dst[feat]=(src[m_idx[feat]]-m_mean[feat]);
		}
	}

	((CDenseFeatures<float64_t>*) features)->set_feature_matrix(dst);
	SG_INFO("new Feature matrix: %ix%i\n", dst.num_rows, dst.num_cols)

	return ((CDenseFeatures<float64_t>*) features)->get_feature_matrix();
}

/// apply preproc on single feature vector
/// result in feature matrix
SGVector<float64_t> CPruneVarSubMean::apply_to_feature_vector(SGVector<float64_t> vector)
{
	ASSERT(m_initialized)

	SGVector<float64_t> out(m_num_idx);

	for (index_t i = 0; i < m_num_idx; ++i)
	{
		out[i] = (vector[m_idx[i]]-m_mean[i]);
		if (m_divide_by_std)
			out[i] /= m_std[i];
	}

	return out;
}

void CPruneVarSubMean::init()
{
	m_initialized = false;
	m_divide_by_std = false;
	m_num_idx = 0;
	m_idx = SGVector<int32_t>();
	m_mean = SGVector<float64_t>();
	m_std = SGVector<float64_t>();
}

void CPruneVarSubMean::register_parameters()
{
	SG_ADD(&m_initialized, "initialized", "The preprocessor is initialized",  MS_NOT_AVAILABLE);
	SG_ADD(&m_divide_by_std, "divide_by_std", "Divide by standard deviation", MS_AVAILABLE);
	SG_ADD(&m_num_idx, "num_idx", "Number of elements in idx_vec", MS_NOT_AVAILABLE);
	SG_ADD(&m_std, "std_vec", "Standard dev vector", MS_NOT_AVAILABLE);
	SG_ADD(&m_mean, "mean_vec", "Mean vector", MS_NOT_AVAILABLE);
	SG_ADD(&m_idx, "idx_vec", "Index vector", MS_NOT_AVAILABLE);
}
