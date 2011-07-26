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
#include <shogun/preprocessor/SimplePreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CPruneVarSubMean::CPruneVarSubMean(bool divide)
: CSimplePreprocessor<float64_t>(), idx(NULL), mean(NULL),
	std(NULL), num_idx(0), divide_by_std(divide), initialized(false)
{
}

CPruneVarSubMean::~CPruneVarSubMean()
{
	cleanup();
}

/// initialize preprocessor from features
bool CPruneVarSubMean::init(CFeatures* features)
{
	if (!initialized)
	{
		ASSERT(features->get_feature_class()==C_SIMPLE);
		ASSERT(features->get_feature_type()==F_DREAL);

		CSimpleFeatures<float64_t>* simple_features=(CSimpleFeatures<float64_t>*) features;
		int32_t num_examples = simple_features->get_num_vectors();
		int32_t num_features = simple_features->get_num_features();

		SG_FREE(mean);
		SG_FREE(idx);
		SG_FREE(std);
		mean=NULL;
		idx=NULL;
		std=NULL;

		mean=SG_MALLOC(float64_t, num_features);
		float64_t* var=SG_MALLOC(float64_t, num_features);
		int32_t i,j;

		for (i=0; i<num_features; i++)
		{
			mean[i]=0;
			var[i]=0 ;
		}

		SGMatrix<float64_t> feature_matrix = simple_features->get_feature_matrix();

		// compute mean
		for (i=0; i<num_examples; i++)
		{
			for (j=0; j<num_features; j++)
				mean[j]+=feature_matrix.matrix[i*num_features+j];
		}

		for (j=0; j<num_features; j++)
			mean[j]/=num_examples;

		// compute var
		for (i=0; i<num_examples; i++)
		{
			for (j=0; j<num_features; j++)
				var[j]+=CMath::sq(mean[j]-feature_matrix.matrix[i*num_features+j]);
		}

		int32_t num_ok=0;
		int32_t* idx_ok=SG_MALLOC(int, num_features);

		for (j=0; j<num_features; j++)
		{
			var[j]/=num_examples;

			if (var[j]>=1e-14) 
			{
				idx_ok[num_ok]=j;
				num_ok++ ;
			}
		}

		SG_INFO( "Reducing number of features from %i to %i\n", num_features, num_ok) ;

		SG_FREE(idx);
		idx=SG_MALLOC(int, num_ok);
		float64_t* new_mean=SG_MALLOC(float64_t, num_ok);
		std=SG_MALLOC(float64_t, num_ok);

		for (j=0; j<num_ok; j++)
		{
			idx[j]=idx_ok[j] ;
			new_mean[j]=mean[idx_ok[j]];
			std[j]=sqrt(var[idx_ok[j]]);
		}
		num_idx = num_ok ;
		SG_FREE(idx_ok);
		SG_FREE(mean);
		SG_FREE(var);
		mean = new_mean;

		initialized = true;
		return true;
	}
	else
		return false;
}

/// clean up allocated memory
void CPruneVarSubMean::cleanup()
{
	SG_FREE(idx);
	idx=NULL;
	SG_FREE(mean);
	mean=NULL;
	SG_FREE(std);
	std=NULL;
}

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
SGMatrix<float64_t> CPruneVarSubMean::apply_to_feature_matrix(CFeatures* features)
{
	ASSERT(initialized);

	int32_t num_vectors=0;
	int32_t num_features=0;
	float64_t* m=((CSimpleFeatures<float64_t>*) features)->get_feature_matrix(num_features, num_vectors);

	SG_INFO( "get Feature matrix: %ix%i\n", num_vectors, num_features);
	SG_INFO( "Preprocessing feature matrix\n");
	for (int32_t vec=0; vec<num_vectors; vec++)
	{
		float64_t* v_src=&m[num_features*vec];
		float64_t* v_dst=&m[num_idx*vec];

		if (divide_by_std)
		{
			for (int32_t feat=0; feat<num_idx; feat++)
				v_dst[feat]=(v_src[idx[feat]]-mean[feat])/std[feat];
		}
		else
		{
			for (int32_t feat=0; feat<num_idx; feat++)
				v_dst[feat]=(v_src[idx[feat]]-mean[feat]);
		}
	}

	((CSimpleFeatures<float64_t>*) features)->set_num_features(num_idx);
	((CSimpleFeatures<float64_t>*) features)->get_feature_matrix(num_features, num_vectors);
	SG_INFO( "new Feature matrix: %ix%i\n", num_vectors, num_features);

	return ((CSimpleFeatures<float64_t>*) features)->get_feature_matrix();
}

/// apply preproc on single feature vector
/// result in feature matrix
SGVector<float64_t> CPruneVarSubMean::apply_to_feature_vector(SGVector<float64_t> vector)
{
	float64_t* ret=NULL;

	if (initialized)
	{
		ret=SG_MALLOC(float64_t, num_idx);

		if (divide_by_std)
		{
			for (int32_t i=0; i<num_idx; i++)
				ret[i]=(vector.vector[idx[i]]-mean[i])/std[i];
		}
		else
		{
			for (int32_t i=0; i<num_idx; i++)
				ret[i]=(vector.vector[idx[i]]-mean[i]);
		}
	}
	else
	{
		ret=SG_MALLOC(float64_t, vector.vlen);
		for (int32_t i=0; i<vector.vlen; i++)
			ret[i]=vector.vector[i];
	}

	return SGVector<float64_t>(ret,num_idx);
}
