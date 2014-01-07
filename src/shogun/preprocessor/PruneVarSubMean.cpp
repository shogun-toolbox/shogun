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

#include <preprocessor/PruneVarSubMean.h>
#include <preprocessor/DensePreprocessor.h>
#include <features/Features.h>
#include <io/SGIO.h>
#include <mathematics/Math.h>

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
		int32_t num_examples = simple_features->get_num_vectors();
		int32_t num_features = simple_features->get_num_features();

		m_mean = SGVector<float64_t>();
		m_idx = SGVector<int32_t>();
		m_std = SGVector<float64_t>();;

		m_mean.resize_vector(num_features);
		float64_t* var=SG_MALLOC(float64_t, num_features);
		int32_t i,j;

		memset(var, 0, num_features*sizeof(float64_t));
		m_mean.zero();

		SGMatrix<float64_t> feature_matrix = simple_features->get_feature_matrix();

		// compute mean
		for (i=0; i<num_examples; i++)
		{
			for (j=0; j<num_features; j++)
				m_mean[j]+=feature_matrix.matrix[i*num_features+j];
		}

		for (j=0; j<num_features; j++)
			m_mean[j]/=num_examples;

		// compute var
		for (i=0; i<num_examples; i++)
		{
			for (j=0; j<num_features; j++)
				var[j]+=CMath::sq(m_mean[j]-feature_matrix.matrix[i*num_features+j]);
		}

		int32_t num_ok=0;
		int32_t* idx_ok=SG_MALLOC(int32_t, num_features);

		for (j=0; j<num_features; j++)
		{
			var[j]/=num_examples;

			if (var[j]>=1e-14)
			{
				idx_ok[num_ok]=j;
				num_ok++ ;
			}
		}

		SG_INFO("Reducing number of features from %i to %i\n", num_features, num_ok)

		m_idx.resize_vector(num_ok);
		SGVector<float64_t> new_mean(num_ok);
		m_std.resize_vector(num_ok);

		for (j=0; j<num_ok; j++)
		{
			m_idx[j]=idx_ok[j] ;
			new_mean[j]=m_mean[idx_ok[j]];
			m_std[j]=CMath::sqrt(var[idx_ok[j]]);
		}
		m_num_idx = num_ok;
		SG_FREE(idx_ok);
		SG_FREE(var);
		m_mean = new_mean;

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

	int32_t num_vectors=0;
	int32_t num_features=0;
	float64_t* m=((CDenseFeatures<float64_t>*) features)->get_feature_matrix(num_features, num_vectors);

	SG_INFO("get Feature matrix: %ix%i\n", num_vectors, num_features)
	SG_INFO("Preprocessing feature matrix\n")
	for (int32_t vec=0; vec<num_vectors; vec++)
	{
		float64_t* v_src=&m[num_features*vec];
		float64_t* v_dst=&m[m_num_idx*vec];

		if (m_divide_by_std)
		{
			for (int32_t feat=0; feat<m_num_idx; feat++)
				v_dst[feat]=(v_src[m_idx[feat]]-m_mean[feat])/m_std[feat];
		}
		else
		{
			for (int32_t feat=0; feat<m_num_idx; feat++)
				v_dst[feat]=(v_src[m_idx[feat]]-m_mean[feat]);
		}
	}

	((CDenseFeatures<float64_t>*) features)->set_num_features(m_num_idx);
	((CDenseFeatures<float64_t>*) features)->get_feature_matrix(num_features, num_vectors);
	SG_INFO("new Feature matrix: %ix%i\n", num_vectors, num_features)

	return ((CDenseFeatures<float64_t>*) features)->get_feature_matrix();
}

/// apply preproc on single feature vector
/// result in feature matrix
SGVector<float64_t> CPruneVarSubMean::apply_to_feature_vector(SGVector<float64_t> vector)
{
	float64_t* ret=NULL;

	if (m_initialized)
	{
		ret=SG_MALLOC(float64_t, m_num_idx);

		if (m_divide_by_std)
		{
			for (int32_t i=0; i<m_num_idx; i++)
				ret[i]=(vector.vector[m_idx[i]]-m_mean[i])/m_std[i];
		}
		else
		{
			for (int32_t i=0; i<m_num_idx; i++)
				ret[i]=(vector.vector[m_idx[i]]-m_mean[i]);
		}
	}
	else
	{
		ret=SG_MALLOC(float64_t, vector.vlen);
		for (int32_t i=0; i<vector.vlen; i++)
			ret[i]=vector.vector[i];
	}

	return SGVector<float64_t>(ret,m_num_idx);
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
	SG_ADD(&m_initialized, "initialized", "The prerpocessor is initialized",  MS_NOT_AVAILABLE);
	SG_ADD(&m_divide_by_std, "divide_by_std", "Divide by standard deviation", MS_AVAILABLE);
	SG_ADD(&m_num_idx, "num_idx", "Number of elements in idx_vec", MS_NOT_AVAILABLE);
	SG_ADD(&m_std, "std_vec", "Standard dev vector", MS_NOT_AVAILABLE);
	SG_ADD(&m_mean, "mean_vec", "Mean vector", MS_NOT_AVAILABLE);
	SG_ADD(&m_idx, "idx_vec", "Index vector", MS_NOT_AVAILABLE);
}
