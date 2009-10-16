/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"
#include "lib/Mathematics.h"

#include <string.h>
#include <stdlib.h>

#ifdef HAVE_LAPACK
#include "lib/lapack.h"

#include "lib/common.h"
#include "preproc/PCACut.h"
#include "preproc/SimplePreProc.h"
#include "features/Features.h"
#include "features/SimpleFeatures.h"
#include "lib/io.h"

using namespace shogun;

CPCACut::CPCACut(int32_t do_whitening_, float64_t thresh_)
: CSimplePreProc<float64_t>("PCACut", "PCAC"), T(NULL), num_dim(0), mean(NULL),
	initialized(false), do_whitening(do_whitening_), thresh(thresh_)
{
}

CPCACut::~CPCACut()
{
	delete[] T;
	delete[] mean;
}

/// initialize preprocessor from features
bool CPCACut::init(CFeatures* f)
{
	if (!initialized)
	{
		ASSERT(f->get_feature_class()==C_SIMPLE);
		ASSERT(f->get_feature_type()==F_DREAL);

		SG_INFO("calling CPCACut::init\n") ;
		int32_t num_vectors=((CSimpleFeatures<float64_t>*)f)->get_num_vectors() ;
		int32_t num_features=((CSimpleFeatures<float64_t>*)f)->get_num_features() ;
		SG_INFO("num_examples: %ld num_features: %ld \n", num_vectors, num_features);
		delete[] mean ;
		mean=new float64_t[num_features+1] ;

		int32_t i,j;

		/// compute mean

		// clear
		for (j=0; j<num_features; j++)
			mean[j]=0 ; 

		// sum 
		for (i=0; i<num_vectors; i++)
		{
			int32_t len;
			bool free;
			float64_t* vec=((CSimpleFeatures<float64_t>*) f)->get_feature_vector(i, len, free);
			for (j=0; j<num_features; j++)
				mean[j]+= vec[j];

			((CSimpleFeatures<float64_t>*) f)->free_feature_vector(vec, i, free);
		}

		//divide
		for (j=0; j<num_features; j++)
			mean[j]/=num_vectors;

		SG_DONE();
		SG_DEBUG("Computing covariance matrix... of size %.2f M\n", num_features*num_features/1024.0/1024.0);
		float64_t *cov=new float64_t[num_features*num_features];

		for (j=0; j<num_features*num_features; j++)
			cov[j]=0.0 ;

		for (i=0; i<num_vectors; i++)
		{
			if (!(i % (num_vectors/10+1)))
				SG_PROGRESS(i, 0, num_vectors);

			int32_t len;
			bool free;

			float64_t* vec=((CSimpleFeatures<float64_t>*) f)->get_feature_vector(i, len, free) ;

			for (int32_t jj=0; jj<num_features; jj++)
				vec[jj]-=mean[jj] ;

			/// A = 1.0*xy^T+A blas
			int nf = (int) num_features; /* calling external lib */
			double* vec_double = (double*) vec; /* calling external lib */
			cblas_dger(CblasColMajor, nf, nf, 1.0, vec_double, 1, vec_double,
				1, (double*) cov, nf);

			//for (int32_t k=0; k<num_features; k++)
			//	for (int32_t l=0; l<num_features; l++)
			//          cov[k*num_features+l]+=feature[l]*feature[k] ;

			((CSimpleFeatures<float64_t>*) f)->free_feature_vector(vec, i, free) ;
		}

		SG_DONE();

		for (i=0; i<num_features; i++)
			for (j=0; j<num_features; j++)
				cov[i*num_features+j]/=num_vectors ;

		SG_DONE();

		SG_INFO("Computing Eigenvalues ... ") ;
		char V='V';
		char U='U';
		int32_t info;
		int32_t ord= num_features;
		int32_t lda= num_features;
		float64_t* eigenvalues=new float64_t[num_features] ;

		for (i=0; i<num_features; i++)
			eigenvalues[i]=0;

		// lapack sym matrix eigenvalues+vectors
		wrap_dsyev(V, U, (int) ord, (double*) cov, (int) lda,
			(double*) eigenvalues, (int*) &info);


		num_dim=0;
		for (i=0; i<num_features; i++)
		{
			//	  SG_DEBUG( "EV[%i]=%e\n", i, values[i]) ;
			if (eigenvalues[i]>thresh)
				num_dim++ ;
		} ;

		SG_INFO("Done\nReducing from %i to %i features..", num_features, num_dim) ;

		delete[] T;
		T=new float64_t[num_dim*num_features];
		num_old_dim=num_features;

		if (do_whitening)
		{
			int32_t offs=0 ;
			for (i=0; i<num_features; i++)
			{
				if (eigenvalues[i]>thresh)
				{
					for (int32_t jj=0; jj<num_features; jj++)
						T[offs+jj*num_dim]=cov[num_features*i+jj]/sqrt(eigenvalues[i]) ;
					offs++ ;
				} ;
			}
		} ;

		delete[] eigenvalues;
		delete[] cov;
		initialized=true;
		SG_INFO("Done\n") ;
		return true ;
	}
	return 
		false;
}

/// initialize preprocessor from features
void CPCACut::cleanup()
{
	delete[] T ;
	T=NULL ;
}

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
float64_t* CPCACut::apply_to_feature_matrix(CFeatures* f)
{
	int32_t num_vectors=0;
	int32_t num_features=0;

	float64_t* m=((CSimpleFeatures<float64_t>*) f)->get_feature_matrix(num_features, num_vectors);
	SG_INFO("get Feature matrix: %ix%i\n", num_vectors, num_features) ;

	if (m)
	{
		SG_INFO("Preprocessing feature matrix\n");
		float64_t* res= new float64_t[num_dim];
		float64_t* sub_mean= new float64_t[num_features];

		for (int32_t vec=0; vec<num_vectors; vec++)
		{
			int32_t i;

			for (i=0; i<num_features; i++)
				sub_mean[i]=m[num_features*vec+i]-mean[i] ;

			int nd = (int) num_dim; /* calling external lib */
			cblas_dgemv(CblasColMajor, CblasNoTrans, nd, (int) num_features,
				1.0, T, nd, (double*) sub_mean, 1, 0, (double*) res, 1);

			float64_t* m_transformed=&m[num_dim*vec];
			for (i=0; i<num_dim; i++)
				m_transformed[i]=m[i];
		}
		delete[] res;
		delete[] sub_mean;

		((CSimpleFeatures<float64_t>*) f)->set_num_features(num_dim);
		((CSimpleFeatures<float64_t>*) f)->get_feature_matrix(num_features, num_vectors);
		SG_INFO("new Feature matrix: %ix%i\n", num_vectors, num_features);
	}

	return m;
}

/// apply preproc on single feature vector
/// result in feature matrix
float64_t* CPCACut::apply_to_feature_vector(float64_t* f, int32_t &len)
{
	float64_t *ret=new float64_t[num_dim];
	float64_t *sub_mean=new float64_t[len];
	for (int32_t i=0; i<len; i++)
		sub_mean[i]=f[i]-mean[i];

	int nd = (int) num_dim;  /* calling external lib */
	cblas_dgemv(CblasColMajor, CblasNoTrans, nd, (int) len, 1.0, (double*) T,
		nd, (double*) sub_mean, 1, 0, (double*) ret, 1);

	delete[] sub_mean ;
	len=num_dim ;
	//	  SG_DEBUG( "num_dim: %d\n", num_dim);
	return ret;
}
#endif
