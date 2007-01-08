/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Gunnar Raetsch
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "preproc/PruneVarSubMean.h"
#include "preproc/RealPreProc.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "lib/io.h"
#include "lib/Mathematics.h"

CPruneVarSubMean::CPruneVarSubMean(bool divide)
  : CRealPreProc("PruneVarSubMean","PVSM"), idx(NULL), mean(NULL), std(NULL), num_idx(0), divide_by_std(divide), initialized(false)
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
		ASSERT(p_f->get_feature_class() == C_SIMPLE);
		ASSERT(p_f->get_feature_type() == F_DREAL);

		CRealFeatures *f=(CRealFeatures*) p_f ;
		INT num_examples=f->get_num_vectors() ;
		INT num_features=((CRealFeatures*)f)->get_num_features() ;

		delete[] mean;
		delete[] idx;
		delete[] std; 
		mean=NULL;
		idx=NULL;
		std=NULL;

		mean=new double[num_features] ;
		double* var=new double[num_features] ;
		INT i,j;

		for (i=0; i<num_features; i++)
		{
			mean[i]=0;
			var[i]=0 ;
		}

		// compute mean
		for (i=0; i<num_examples; i++)
		{
			INT len ; bool free ;
			DREAL* feature=f->get_feature_vector(i, len, free) ;

			for (j=0; j<len; j++)
				mean[j]+=feature[j];

			f->free_feature_vector(feature, i, free) ;
		}

		for (j=0; j<num_features; j++)
			mean[j]/=num_examples ;

		// compute var
		for (i=0; i<num_examples; i++)
		{
			INT len ; bool free ;
			DREAL* feature=f->get_feature_vector(i, len, free) ;

			for (j=0; j<num_features; j++)
				var[j]+=(mean[j]-feature[j])*(mean[j]-feature[j]) ;

			f->free_feature_vector(feature, i, free) ;
		}

		INT num_ok=0;
		INT* idx_ok=new int[num_features];

		for (j=0; j<num_features; j++)
		{
			var[j]/=num_examples ;

			if (var[j]>=1e-14) 
			{
				idx_ok[num_ok]=j ;
				num_ok++ ;
			}
		}

		CIO::message(M_INFO, "Reducing number of features from %i to %i\n", num_features, num_ok) ;

		delete[] idx ;
		idx=new int[num_ok];
		DREAL* new_mean=new DREAL[num_ok];
		std=new DREAL[num_ok];

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
DREAL* CPruneVarSubMean::apply_to_feature_matrix(CFeatures* f)
{
	ASSERT(initialized);

    INT num_vectors=0;
    INT num_features=0;
    DREAL* m=((CRealFeatures*) f)->get_feature_matrix(num_features, num_vectors);

    CIO::message(M_INFO, "get Feature matrix: %ix%i\n", num_vectors, num_features) ;
	CIO::message(M_INFO, "Preprocessing feature matrix\n");
    for (INT vec=0; vec<num_vectors; vec++)
	{
		DREAL* v_src=&m[num_features*vec];
		DREAL* v_dst=&m[num_idx*vec];

		if (divide_by_std)
		{
			for (INT feat=0; feat<num_idx; feat++)
				v_dst[feat]=(v_src[idx[feat]]-mean[feat])/std[feat];
		}
		else
		{
			for (INT feat=0; feat<num_idx; feat++)
				v_dst[feat]=(v_src[idx[feat]]-mean[feat]);
		}
	}
	
	((CRealFeatures*) f)->set_num_features(num_idx);
	((CRealFeatures*) f)->get_feature_matrix(num_features, num_vectors);
	CIO::message(M_INFO, "new Feature matrix: %ix%i\n", num_vectors, num_features);
    
    return m;
}

/// apply preproc on single feature vector
/// result in feature matrix
DREAL* CPruneVarSubMean::apply_to_feature_vector(DREAL* f, INT &len)
{
	DREAL* ret=NULL;

  if (initialized)
  {
	  ret=new DREAL[num_idx] ;

	  if (divide_by_std)
	  {
		  for (INT i=0; i<num_idx; i++)
			  ret[i]=(f[idx[i]]-mean[i])/std[i];
	  }
	  else
	  {
		  for (INT i=0; i<num_idx; i++)
			  ret[i]=(f[idx[i]]-mean[i]);
	  }
	  len=num_idx ;
  }
  else
  {
	  ret=new DREAL[len] ;
	  for (INT i=0; i<len; i++)
		  ret[i]=f[i];
  }

  return ret;
}

/// initialize preprocessor from file
bool CPruneVarSubMean::load_init_data(FILE* src)
{
	bool result=false;
	INT divide=0;
	

    ASSERT(fread(&divide, sizeof(int), 1, src)==1) ;
    ASSERT(fread(&num_idx, sizeof(int), 1, src)==1) ;
	CIO::message(M_INFO, "divide:%d num_idx:%d\n", divide, num_idx);
	delete[] mean;
	delete[] idx;
	delete[] std;
	idx=new int[num_idx];
	mean=new DREAL[num_idx];
	std=new DREAL[num_idx];
	ASSERT (mean!=NULL && idx!=NULL && std!=NULL);
    ASSERT(fread(idx, sizeof(int), num_idx, src)==(UINT) num_idx) ;
    ASSERT(fread(mean, sizeof(DREAL), num_idx, src)==(UINT) num_idx) ;
    ASSERT(fread(std, sizeof(DREAL), num_idx, src)==(UINT) num_idx) ;
	result=true;
	divide_by_std=(divide==1);
	initialized=true;
	return result;
}

/// save init-data (like transforamtion matrices etc) to file
bool CPruneVarSubMean::save_init_data(FILE* dst)
{
	return false;
}
