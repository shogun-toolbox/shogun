/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
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
#include "features/RealFeatures.h"
#include "lib/io.h"

CPCACut::CPCACut(INT do_whitening_, double thresh_)
: CSimplePreProc<DREAL>("PCACut", "PCAC"), T(NULL), num_dim(0), mean(NULL),
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
		ASSERT(f->get_feature_class() == C_SIMPLE);
		ASSERT(f->get_feature_type() == F_DREAL);

		SG_INFO("calling CPCACut::init\n") ;
		INT num_vectors=((CRealFeatures*)f)->get_num_vectors() ;
		INT num_features=((CRealFeatures*)f)->get_num_features() ;
		SG_INFO("num_examples: %ld num_features: %ld \n", num_vectors, num_features);
		delete[] mean ;
		mean=new double[num_features+1] ;

		INT i,j;

		/// compute mean

		// clear
		for (j=0; j<num_features; j++)
		{
			mean[j]=0 ; 
		}

		// sum 
		for (i=0; i<num_vectors; i++)
		{
			INT len;
			bool free;
			DREAL* vec=((CRealFeatures*) f)->get_feature_vector(i, len, free);
			for (j=0; j<num_features; j++)
			{
				mean[j]+= vec[j];
			}
			((CRealFeatures*) f)->free_feature_vector(vec, i, free);
		}

		//divide
		for (j=0; j<num_features; j++)
			mean[j]/=num_vectors;

		SG_INFO("done.\nComputing covariance matrix... of size %.2f M\n", num_features*num_features/1024.0/1024.0) ;
		double *cov=new double[num_features*num_features] ;
		ASSERT(cov!=NULL) ;

		for (j=0; j<num_features*num_features; j++)
			cov[j]=0.0 ;

		for (i=0; i<num_vectors; i++)
		{
			if (!(i % (num_vectors/10+1)))
				SG_PROGRESS(i, 0, num_vectors);

			INT len;
			bool free;

			DREAL* vec=((CRealFeatures*) f)->get_feature_vector(i, len, free) ;

			for (INT jj=0; jj<num_features; jj++)
				vec[jj]-=mean[jj] ;

			/// A = 1.0*xy^T+A blas
			cblas_dger(CblasColMajor, num_features,num_features, 1.0, vec, 1, 
					vec, 1, cov, (int)num_features) ;

			//for (INT k=0; k<num_features; k++)
			//	for (INT l=0; l<num_features; l++)
			//          cov[k*num_features+l]+=feature[l]*feature[k] ;

			((CRealFeatures*) f)->free_feature_vector(vec, i, free) ;
		}

		SG_PRINT( "done.           \n");

		for (i=0; i<num_features; i++)
			for (j=0; j<num_features; j++)
				cov[i*num_features+j]/=num_vectors ;

		SG_INFO("done\n") ;

		SG_INFO("Computing Eigenvalues ... ") ;
		char V='V';
		char U='U';
		//#ifdef DARWIN
		//		__CLPK_integer ord= (int) num_features;
		//		__CLPK_integer lda= (int) num_features;
		//		__CLPK_integer info;
		//		__CLPK_integer lwork=3*num_features ;
		//		__CLPK_doublereal* work=new __CLPK_doublereal[lwork] ;
		//		__CLPK_doublereal* eigenvalues=new __CLPK_doublereal[num_features] ;
		//#else
		int info;
		int ord= (int) num_features;
		int lda= (int) num_features;
		double* eigenvalues=new double[num_features] ;
		//#endif

		for (i=0; i<num_features; i++)
			eigenvalues[i]=0;

		// lapack sym matrix eigenvalues+vectors
		wrap_dsyev(V, U, ord, cov, lda, eigenvalues, &info);

		num_dim=0;
		for (i=0; i<num_features; i++)
		{
			//	  SG_DEBUG( "EV[%i]=%e\n", i, values[i]) ;
			if (eigenvalues[i]>thresh)
				num_dim++ ;
		} ;

		SG_INFO("Done\nReducing from %i to %i features..", num_features, num_dim) ;

		delete[] T;
		T=new DREAL[num_dim*num_features] ;
		num_old_dim=num_features;

		ASSERT(T!=NULL) ;
		if (do_whitening)
		{
			INT offs=0 ;
			for (i=0; i<num_features; i++)
			{
				if (eigenvalues[i]>1e-6)
				{
					for (INT jj=0; jj<num_features; jj++)
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
DREAL* CPCACut::apply_to_feature_matrix(CFeatures* f)
{
	INT num_vectors=0;
	INT num_features=0;

	DREAL* m=((CRealFeatures*) f)->get_feature_matrix(num_features, num_vectors);
	SG_INFO("get Feature matrix: %ix%i\n", num_vectors, num_features) ;

	if (m)
	{
		SG_INFO("Preprocessing feature matrix\n");
		DREAL* res= new DREAL[num_dim];
		double* sub_mean= new double[num_features];

		for (INT vec=0; vec<num_vectors; vec++)
		{
			INT i;

			for (i=0; i<num_features; i++)
				sub_mean[i]=m[num_features*vec+i]-mean[i] ;

			cblas_dgemv(CblasColMajor, CblasNoTrans, num_dim, num_features, 1.0,
					T, num_dim, sub_mean, 1, 0, res, 1); 

			DREAL* m_transformed=&m[num_dim*vec];
			for (i=0; i<num_dim; i++)
				m_transformed[i]=m[i];
		}
		delete[] res;
		delete[] sub_mean;

		((CRealFeatures*) f)->set_num_features(num_dim);
		((CRealFeatures*) f)->get_feature_matrix(num_features, num_vectors);
		SG_INFO("new Feature matrix: %ix%i\n", num_vectors, num_features);
	}

	return m;
}

/// apply preproc on single feature vector
/// result in feature matrix
DREAL* CPCACut::apply_to_feature_vector(DREAL* f, INT &len)
{
	DREAL *ret=new DREAL[num_dim];
	DREAL *sub_mean=new DREAL[len];
	for (INT i=0; i<len; i++)
		sub_mean[i]=f[i]-mean[i];

	cblas_dgemv(CblasColMajor, CblasNoTrans, num_dim, len, 1.0 , T, num_dim, sub_mean, 1, 0, ret, 1) ;
	//void cblas_dgemv(const enum CBLAS_ORDER order,
	//                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
	//                 const double alpha, const double *A, const int lda,
	//                 const double *X, const int incX, const double beta,
	//                 double *Y, const int incY);
	//

	delete[] sub_mean ;
	len=num_dim ;
	//	  SG_DEBUG( "num_dim: %d\n", num_dim);
	return ret;
}

/// initialize preprocessor from file
bool CPCACut::load_init_data(FILE* src)
{
	ASSERT(fread(&num_dim, sizeof(int), 1, src)==1);
	ASSERT(fread(&num_old_dim, sizeof(int), 1, src)==1);
	delete[] mean;
	delete[] T;
	mean=new double[num_dim];
	T=new double[num_dim*num_old_dim];
	ASSERT (mean!=NULL && T!=NULL);
	ASSERT(fread(mean, sizeof(double), num_old_dim, src)==(UINT) num_old_dim);
	ASSERT(fread(T, sizeof(double), num_dim*num_old_dim, src)==(UINT) num_old_dim*num_dim);
	return true;
}

/// save init-data (like transforamtion matrices etc) to file
bool CPCACut::save_init_data(FILE* dst)
{
	ASSERT(fwrite(&num_dim, sizeof(int), 1, dst)==1);
	ASSERT(fwrite(&num_old_dim, sizeof(int), 1, dst)==1);
	ASSERT(fwrite(mean, sizeof(double), num_old_dim, dst)==(UINT) num_old_dim);
	ASSERT(fwrite(T, sizeof(double), num_dim*num_old_dim, dst)==(UINT) num_old_dim*num_dim);
	return true;
}
#endif // HAVE_LAPACK
