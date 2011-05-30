/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 1999-2008,2011 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2011 Berlin Institute of Technology
 */

#include "lib/config.h"
#include "lib/Mathematics.h"

#include <string.h>
#include <stdlib.h>

#ifdef HAVE_LAPACK
#include "lib/lapack.h"

#include "lib/common.h"
#include "preprocessor/PCACut.h"
#include "preprocessor/SimplePreprocessor.h"
#include "features/Features.h"
#include "features/SimpleFeatures.h"
#include "lib/io.h"

using namespace shogun;

CPCACut::CPCACut(bool do_whitening_, ECutoffType cutoff_type_, float64_t thresh_)
: CSimplePreprocessor<float64_t>(), T(NULL), num_dim(0), mean(NULL),
	length_mean(NULL), eigenvalues(NULL), num_eigenvalues(0),initialized(false),
	do_whitening(do_whitening_), cutoff_type(cutoff_type_), thresh(thresh_)
{
	init();
}

CPCACut::~CPCACut()
{
	delete[] T;
	delete[] mean;
	delete[] eigenvalues;
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
		mean=new float64_t[num_features];
		length_mean=num_features;

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

		float64_t* sub_mean= new float64_t[num_features];
		for (i=0; i<num_vectors; i++)
		{
			if (!(i % (num_vectors/10+1)))
				SG_PROGRESS(i, 0, num_vectors);

			int32_t len;
			bool free;

			float64_t* vec=((CSimpleFeatures<float64_t>*) f)->get_feature_vector(i, len, free) ;
            for (int32_t jj=0; jj<num_features; jj++)
                sub_mean[jj]=vec[jj]-mean[jj];

            /// A = 1.0*xy^T+A blas
            int nf = (int) num_features; /* calling external lib */
            cblas_dger(CblasColMajor, nf, nf, 1.0, sub_mean, 1, sub_mean,
                    1, (double*) cov, nf);

            ((CSimpleFeatures<float64_t>*) f)->free_feature_vector(vec, i, free) ;
		}
        delete[] sub_mean;

		SG_DONE();

		for (i=0; i<num_features; i++)
		{
			for (j=0; j<num_features; j++)
				cov[i*num_features+j]/=(num_vectors-1);
		}


		SG_INFO("Computing Eigenvalues ... ") ;
		eigenvalues=CMath::compute_eigenvectors(cov, num_features, num_features);
        num_eigenvalues=num_features;

		num_dim=0;
		if (cutoff_type == FIXED_NUMBER)
		{
			ASSERT(thresh <= num_features);
			num_dim = thresh;
		}
		else if (cutoff_type == VARIANCE_EXPLAINED)
		{
			float64_t eig_sum = 0;
			for (i=0; i<num_features; i++)
				eig_sum += eigenvalues[i];
			
			float64_t com_sum = 0;		
			for (i=num_features-1; i>-1; i--)
			{
				num_dim++;
				com_sum += eigenvalues[i];
				if (com_sum/eig_sum>=thresh)
					break;
			}
		}
		else
		{
			for (i=num_features-1; i>-1; i--)
			{
				if (eigenvalues[i]>thresh)
					num_dim++;
				else
					break;
			}
		}

		SG_INFO("Done\nReducing from %i to %i features..", num_features, num_dim) ;

		delete[] T;
		T=new float64_t[num_dim*num_features];
		num_old_dim=num_features;

		int32_t offs=0 ;
		for (i=num_features-1; i<num_features-num_dim-1; i++)
		{
			for (int32_t jj=0; jj<num_features; jj++)
				if (do_whitening)
					T[offs+jj*num_dim]=cov[num_features*i+jj]/sqrt(eigenvalues[i]);
				else
					T[offs+jj*num_dim]=cov[num_features*i+jj];
			offs++;
		}

		delete[] cov;
		initialized=true;
		return true;
	}

	return false;
}

/// initialize preprocessor from features
void CPCACut::cleanup()
{
	delete[] T;
	T=NULL;
	num_dim=0;
	num_old_dim=0;

	delete[] mean;
	length_mean=0;
}

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
float64_t* CPCACut::apply_to_feature_matrix(CFeatures* f)
{
	int32_t num_vectors=0;
	int32_t num_features=0;

	float64_t* m=((CSimpleFeatures<float64_t>*) f)->get_feature_matrix(num_features, num_vectors);
	SG_INFO("get Feature matrix: %ix%i\n", num_vectors, num_features);

	if (m)
	{
		SG_INFO("Preprocessing feature matrix\n");
		float64_t* res= new float64_t[num_dim];
		float64_t* sub_mean= new float64_t[num_features];

		for (int32_t vec=0; vec<num_vectors; vec++)
		{
			int32_t i;

			for (i=0; i<num_features; i++)
				sub_mean[i]=m[num_features*vec+i]-mean[i];

			cblas_dgemv(CblasColMajor, CblasNoTrans, num_dim, (int) num_features,
				1.0, T, num_dim, (double*) sub_mean, 1, 0, (double*) res, 1);

			float64_t* m_transformed=&m[num_dim*vec];
			for (i=0; i<num_dim; i++)
				m_transformed[i]=res[i];
		}
		delete[] res;
		delete[] sub_mean;

		((CSimpleFeatures<float64_t>*) f)->set_num_features(num_dim);
		((CSimpleFeatures<float64_t>*) f)->get_feature_matrix(num_features, num_vectors);
		SG_INFO("new Feature matrix: %ix%i\n", num_vectors, num_features);
	}

	return m;
}

/// apply preproc to single feature vector
float64_t* CPCACut::apply_to_feature_vector(float64_t* f, int32_t &len)
{
	float64_t *ret=new float64_t[num_dim];
	float64_t *sub_mean=new float64_t[len];
	for (int32_t i=0; i<len; i++)
		sub_mean[i]=f[i]-mean[i];

	int nd = (int) num_dim;  /* calling external lib */
	cblas_dgemv(CblasColMajor, CblasNoTrans, nd, (int) len, 1.0, (double*) T,
		nd, (double*) sub_mean, 1, 0, (double*) ret, 1);

	delete[] sub_mean;
	len=num_dim;
	return ret;
}

void CPCACut::get_transformation_matrix(float64_t** dst, int32_t* num_feat, int32_t* num_new_dim)
{
	ASSERT(T);

	int64_t num=int64_t(num_dim)*num_old_dim;
	*num_feat=num_old_dim;
	*num_new_dim=num_dim;
	*dst=(float64_t*) SG_MALLOC(sizeof(float64_t)*num);
	memcpy(*dst, T, num * sizeof(float64_t));
}

void CPCACut::get_eigenvalues(float64_t** dst, int32_t* new_num_dim)
{
	ASSERT(eigenvalues);

	*new_num_dim=num_eigenvalues;
	*dst=(float64_t*) SG_MALLOC(sizeof(float64_t)*num_eigenvalues);
	memcpy(*dst, eigenvalues, num_eigenvalues * sizeof(float64_t));
}

void CPCACut::get_mean(float64_t** dst, int32_t* num_feat)
{
	ASSERT(mean);

	*num_feat=length_mean;
	*dst=(float64_t*) SG_MALLOC(sizeof(float64_t)*length_mean);
	memcpy(*dst, mean, length_mean * sizeof(float64_t));
}

void CPCACut::init()
{
	m_parameters->add_matrix(&T, &num_dim, &num_old_dim,
					"T", "Transformation matrix (Eigenvectors of covarience matrix).");
	m_parameters->add_vector(&mean, &length_mean,
					"mean", "Mean Vector.");
	m_parameters->add_vector(&eigenvalues, &num_eigenvalues,
					"eigenvalues", "Vector with Eigenvalues.");
	m_parameters->add(&initialized,
			"initalized", "True when initialized.");
	m_parameters->add(&do_whitening,
			"do_whitening", "Whether data shall be whitened.");
	m_parameters->add((machine_int_t*) &cutoff_type, "cutoff_type",
			"Cutoff type.");
	m_parameters->add(&thresh,
			"thresh", "Cutoff threshold.");
}
#endif
