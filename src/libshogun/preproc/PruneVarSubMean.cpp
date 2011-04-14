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

#include "preproc/PruneVarSubMean.h"
#include "preproc/SimplePreProc.h"
#include "features/Features.h"
#include "features/SimpleFeatures.h"
#include "lib/io.h"
#include "lib/Mathematics.h"

using namespace shogun;

CPruneVarSubMean::CPruneVarSubMean(bool divide)
: CSimplePreProc<float64_t>(), idx(NULL), mean(NULL),
	std(NULL), num_idx(0), divide_by_std(divide), initialized(false)
{
}

CPruneVarSubMean::~CPruneVarSubMean()
{
	cleanup();
}

/// initialize preprocessor from features
bool CPruneVarSubMean::init(CFeatures* p_f)
{
	if (!initialized)
	{
		ASSERT(p_f->get_feature_class()==C_SIMPLE);
		ASSERT(p_f->get_feature_type()==F_DREAL);

		CSimpleFeatures<float64_t> *f=(CSimpleFeatures<float64_t>*) p_f;
		int32_t num_examples=f->get_num_vectors();
		int32_t num_features=((CSimpleFeatures<float64_t>*)f)->get_num_features();

		delete[] mean;
		delete[] idx;
		delete[] std;
		mean=NULL;
		idx=NULL;
		std=NULL;

		mean=new float64_t[num_features];
		float64_t* var=new float64_t[num_features];
		int32_t i,j;

		for (i=0; i<num_features; i++)
		{
			mean[i]=0;
			var[i]=0 ;
		}

		// compute mean
		for (i=0; i<num_examples; i++)
		{
			int32_t len ; bool free ;
			float64_t* feature=f->get_feature_vector(i, len, free) ;

			for (j=0; j<len; j++)
				mean[j]+=feature[j];

			f->free_feature_vector(feature, i, free) ;
		}

		for (j=0; j<num_features; j++)
			mean[j]/=num_examples ;

		// compute var
		for (i=0; i<num_examples; i++)
		{
			int32_t len ; bool free ;
			float64_t* feature=f->get_feature_vector(i, len, free) ;

			for (j=0; j<num_features; j++)
				var[j]+=(mean[j]-feature[j])*(mean[j]-feature[j]) ;

			f->free_feature_vector(feature, i, free) ;
		}

		int32_t num_ok=0;
		int32_t* idx_ok=new int[num_features];

		for (j=0; j<num_features; j++)
		{
			var[j]/=num_examples ;

			if (var[j]>=1e-14) 
			{
				idx_ok[num_ok]=j ;
				num_ok++ ;
			}
		}

		SG_INFO( "Reducing number of features from %i to %i\n", num_features, num_ok) ;

		delete[] idx ;
		idx=new int[num_ok];
		float64_t* new_mean=new float64_t[num_ok];
		std=new float64_t[num_ok];

		for (j=0; j<num_ok; j++)
		{
			idx[j]=idx_ok[j] ;
			new_mean[j]=mean[idx_ok[j]];
			std[j]=sqrt(var[idx_ok[j]]);
		}
		num_idx=num_ok ;
		delete[] idx_ok ;
		delete[] mean;
		delete[] var;
		mean=new_mean;

		initialized=true;
		return true ;
	}
	else
		return false;
}

/// clean up allocated memory
void CPruneVarSubMean::cleanup()
{
	delete[] idx;
	idx=NULL;
	delete[] mean;
	mean=NULL;
	delete[] std;
	std=NULL;
}

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
float64_t* CPruneVarSubMean::apply_to_feature_matrix(CFeatures* f)
{
	ASSERT(initialized);

	int32_t num_vectors=0;
	int32_t num_features=0;
	float64_t* m=((CSimpleFeatures<float64_t>*) f)->get_feature_matrix(num_features, num_vectors);

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

	((CSimpleFeatures<float64_t>*) f)->set_num_features(num_idx);
	((CSimpleFeatures<float64_t>*) f)->get_feature_matrix(num_features, num_vectors);
	SG_INFO( "new Feature matrix: %ix%i\n", num_vectors, num_features);

	return m;
}

/// apply preproc on single feature vector
/// result in feature matrix
float64_t* CPruneVarSubMean::apply_to_feature_vector(float64_t* f, int32_t &len)
{
	float64_t* ret=NULL;

	if (initialized)
	{
		ret=new float64_t[num_idx] ;

		if (divide_by_std)
		{
			for (int32_t i=0; i<num_idx; i++)
				ret[i]=(f[idx[i]]-mean[i])/std[i];
		}
		else
		{
			for (int32_t i=0; i<num_idx; i++)
				ret[i]=(f[idx[i]]-mean[i]);
		}
		len=num_idx ;
	}
	else
	{
		ret=new float64_t[len] ;
		for (int32_t i=0; i<len; i++)
			ret[i]=f[i];
	}

	return ret;
}
