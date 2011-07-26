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
#include <shogun/preprocessor/PCA.h>
#ifdef HAVE_LAPACK
#include <shogun/mathematics/lapack.h>
#include <shogun/lib/config.h>
#include <shogun/mathematics/Math.h>
#include <string.h>
#include <stdlib.h>
#include <shogun/lib/common.h>
#include <shogun/preprocessor/SimplePreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

CPCA::CPCA(bool do_whitening_, EPCAMode mode_, float64_t thresh_)
: CSimplePreprocessor<float64_t>(), num_dim(0), initialized(false),
	do_whitening(do_whitening_), m_mode(mode_), thresh(thresh_)
{
	m_transformation_matrix = SGMatrix<float64_t>(NULL,0,0,true);
	m_mean_vector = SGVector<float64_t>(NULL,0,true);
	m_eigenvalues_vector = SGVector<float64_t>(NULL,0,true);
	init();
}

CPCA::~CPCA()
{
	m_transformation_matrix.free_matrix();
	m_mean_vector.free_vector();
	m_eigenvalues_vector.free_vector();
}

bool CPCA::init(CFeatures* features)
{
	if (!initialized)
	{
		ASSERT(features->get_feature_class()==C_SIMPLE);
		ASSERT(features->get_feature_type()==F_DREAL);

		int32_t num_vectors=((CSimpleFeatures<float64_t>*)features)->get_num_vectors();
		int32_t num_features=((CSimpleFeatures<float64_t>*)features)->get_num_features();
		SG_INFO("num_examples: %ld num_features: %ld \n", num_vectors, num_features);
		
		m_mean_vector.vlen = num_features;
		m_mean_vector.vector = new float64_t[num_features];
			
		// loop varibles
		int32_t i,j,k;

		for (i=0; i<num_features; i++)
			m_mean_vector.vector[i] = 0.0;

		// sum 
		SGMatrix<float64_t> feature_matrix = ((CSimpleFeatures<float64_t>*)features)->get_feature_matrix();
		for (i=0; i<num_vectors; i++)
		{
			for (j=0; j<num_features; j++)
				m_mean_vector.vector[j] += feature_matrix.matrix[i*num_features+j];
		}

		//divide
		for (j=0; j<num_features; j++)
			m_mean_vector.vector[j] /= num_vectors;

		SG_DONE();
		SG_DEBUG("Computing covariance matrix... of size %.2f M\n", num_features*num_features/1024.0/1024.0);

		float64_t* cov = new float64_t[num_features*num_features];

		for (j=0; j<num_features*num_features; j++)
			cov[j] = 0.0;

		float64_t* sub_mean= new float64_t[num_features];

		for (i=0; i<num_vectors; i++)
		{
			if (!(i % (num_vectors/10+1)))
				SG_PROGRESS(i, 0, num_vectors);

            		for (k=0; k<num_features; k++)
                		sub_mean[k]=feature_matrix.matrix[i*num_features+k]-m_mean_vector.vector[k];

            		cblas_dger(CblasColMajor,
			           num_features,num_features,
			           1.0,sub_mean,1,
			           sub_mean,1,
			           cov, num_features);
		}

        	SG_FREE(sub_mean);

		SG_DONE();

		for (i=0; i<num_features; i++)
		{
			for (j=0; j<num_features; j++)
				cov[i*num_features+j]/=(num_vectors-1);
		}

		SG_INFO("Computing Eigenvalues ... ") ;

		m_eigenvalues_vector.vector = CMath::compute_eigenvectors(cov,num_features,num_features);
		m_eigenvalues_vector.vlen = num_features;
		num_dim=0;

		if (m_mode == FIXED_NUMBER)
		{
			ASSERT(thresh <= num_features);
			num_dim = thresh;
		}
		if (m_mode == VARIANCE_EXPLAINED)
		{
			float64_t eig_sum = 0;
			for (i=0; i<num_features; i++)
				eig_sum += m_eigenvalues_vector.vector[i];
			
			float64_t com_sum = 0;		
			for (i=num_features-1; i>-1; i--)
			{
				num_dim++;
				com_sum += m_eigenvalues_vector.vector[i];
				if (com_sum/eig_sum>=thresh)
					break;
			}
		}
		if (m_mode == THRESHOLD)
		{
			for (i=num_features-1; i>-1; i--)
			{
				if (m_eigenvalues_vector.vector[i]>thresh)
					num_dim++;
				else
					break;
			}
		}

		SG_INFO("Done\nReducing from %i to %i features..", num_features, num_dim) ;
		
		m_transformation_matrix = SGMatrix<float64_t>(num_features,num_dim);
		num_old_dim=num_features;

		int32_t offs=0;
		for (i=num_features-1; i<num_features-num_dim-1; i++)
		{
			for (k=0; k<num_features; k++)
				if (do_whitening)
					m_transformation_matrix.matrix[offs+k*num_dim] =
						cov[num_features*i+k]/sqrt(m_eigenvalues_vector.vector[i]);
				else
					m_transformation_matrix.matrix[offs+k*num_dim] = 
						cov[num_features*i+k];
			offs++;
		}

		SG_FREE(cov);
		initialized = true;
		return true;
	}

	return false;
}

void CPCA::cleanup()
{
	m_transformation_matrix.free_matrix();
	m_mean_vector.free_vector();
	m_eigenvalues_vector.free_vector();
}

SGMatrix<float64_t> CPCA::apply_to_feature_matrix(CFeatures* features)
{
	SGMatrix<float64_t> m = ((CSimpleFeatures<float64_t>*) features)->get_feature_matrix();
	int32_t num_vectors = m.num_cols;
	int32_t num_features = m.num_rows;
	SG_INFO("get Feature matrix: %ix%i\n", num_vectors, num_features);

	if (m.matrix)
	{
		SG_INFO("Preprocessing feature matrix\n");
		float64_t* res = new float64_t[num_dim];
		float64_t* sub_mean = new float64_t[num_features];

		for (int32_t vec=0; vec<num_vectors; vec++)
		{
			int32_t i;

			for (i=0; i<num_features; i++)
				sub_mean[i] = m.matrix[num_features*vec+i] - m_mean_vector.vector[i];

			cblas_dgemv(CblasColMajor,CblasNoTrans, 
			            num_dim,num_features,
			            1.0,m_transformation_matrix.matrix,num_dim,
			            sub_mean,1,
			            0.0,res,1);

			float64_t* m_transformed = &m.matrix[num_dim*vec];

			for (i=0; i<num_dim; i++)
				m_transformed[i] = res[i];
		}
		SG_FREE(res);
		SG_FREE(sub_mean);

		((CSimpleFeatures<float64_t>*) features)->set_num_features(num_dim);
		((CSimpleFeatures<float64_t>*) features)->get_feature_matrix(num_features, num_vectors);
		SG_INFO("new Feature matrix: %ix%i\n", num_vectors, num_features);
	}

	return m;
}

SGVector<float64_t> CPCA::apply_to_feature_vector(SGVector<float64_t> vector)
{
	float64_t* result = new float64_t[num_dim];
	float64_t* sub_mean = new float64_t[vector.vlen];
	
	for (int32_t i=0; i<vector.vlen; i++)
		sub_mean[i]=vector.vector[i]-m_mean_vector.vector[i];

	cblas_dgemv(CblasColMajor,CblasNoTrans,
	            num_dim,vector.vlen, 
	            1.0,m_transformation_matrix.matrix,m_transformation_matrix.num_cols, 
	            sub_mean,1, 
	            0.0,result,1);

	SG_FREE(sub_mean);
	return SGVector<float64_t>(result,num_dim);
}

SGMatrix<float64_t> CPCA::get_transformation_matrix()
{
	return SGMatrix<float64_t>(m_transformation_matrix.matrix,
	                           m_transformation_matrix.num_rows,
	                           m_transformation_matrix.num_cols,
	                           false);
}

SGVector<float64_t> CPCA::get_eigenvalues()
{
	return SGVector<float64_t>(m_eigenvalues_vector.vector,
	                           m_eigenvalues_vector.vlen,
	                           false);
}

SGVector<float64_t> CPCA::get_mean()
{
	return SGVector<float64_t>(m_mean_vector.vector,
	                           m_mean_vector.vlen,
	                           false);
}

void CPCA::init()
{
//	m_parameters->add_matrix(&T, &num_dim, &num_old_dim,
//					"T", "Transformation matrix (Eigenvectors of covarience matrix).");
//	m_parameters->add_vector(&mean, &length_mean,
//					"mean", "Mean Vector.");
//	m_parameters->add_vector(&eigenvalues, &num_eigenvalues,
//					"eigenvalues", "Vector with Eigenvalues.");
	m_parameters->add(&initialized,
			"initalized", "True when initialized.");
	m_parameters->add(&do_whitening,
			"do_whitening", "Whether data shall be whitened.");
	m_parameters->add((machine_int_t*) &m_mode, "mode",
			"PCA Mode.");
	m_parameters->add(&thresh,
			"thresh", "Cutoff threshold.");
}
#endif /* HAVE_LAPACK */
