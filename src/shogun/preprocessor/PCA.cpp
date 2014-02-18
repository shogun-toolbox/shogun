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
#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CPCA::CPCA(bool do_whitening_, EPCAMode mode_, float64_t thresh_)
: CDimensionReductionPreprocessor(), num_dim(0), m_initialized(false),
	m_whitening(do_whitening_), m_mode(mode_), thresh(thresh_)
{
	init();
}

void CPCA::init()
{
	m_transformation_matrix = SGMatrix<float64_t>();
	m_mean_vector = SGVector<float64_t>();
	m_eigenvalues_vector = SGVector<float64_t>();

	SG_ADD(&m_transformation_matrix, "transformation_matrix",
	    "Transformation matrix (Eigenvectors of covariance matrix).",
	    MS_NOT_AVAILABLE);
	SG_ADD(&m_mean_vector, "mean_vector", "Mean Vector.", MS_NOT_AVAILABLE);
	SG_ADD(&m_eigenvalues_vector, "eigenvalues_vector",
	    "Vector with Eigenvalues.", MS_NOT_AVAILABLE);
	SG_ADD(&m_initialized, "initalized", "True when initialized.",
	    MS_NOT_AVAILABLE);
	SG_ADD(&m_whitening, "whitening", "Whether data shall be whitened.",
	    MS_AVAILABLE);
	SG_ADD((machine_int_t*) &m_mode, "mode", "PCA Mode.", MS_AVAILABLE);
	SG_ADD(&thresh, "thresh", "Cutoff threshold.", MS_AVAILABLE);
}

CPCA::~CPCA()
{
}

bool CPCA::init(CFeatures* features)
{
	if (!m_initialized)
	{
		// loop varibles
		int32_t i,j,k;

		ASSERT(features->get_feature_class()==C_DENSE)
		ASSERT(features->get_feature_type()==F_DREAL)

		int32_t num_vectors=((CDenseFeatures<float64_t>*)features)->get_num_vectors();
		int32_t num_features=((CDenseFeatures<float64_t>*)features)->get_num_features();
		SG_INFO("num_examples: %ld num_features: %ld \n", num_vectors, num_features)

		m_mean_vector.vlen = num_features;
		m_mean_vector.vector = SG_CALLOC(float64_t, num_features);

		// sum
		SGMatrix<float64_t> feature_matrix = ((CDenseFeatures<float64_t>*)features)->get_feature_matrix();
		for (i=0; i<num_vectors; i++)
		{
			for (j=0; j<num_features; j++)
				m_mean_vector.vector[j] += feature_matrix.matrix[i*num_features+j];
		}

		//divide
		for (i=0; i<num_features; i++)
			m_mean_vector.vector[i] /= num_vectors;

		float64_t* cov = SG_CALLOC(float64_t, num_features*num_features);

		float64_t* sub_mean = SG_MALLOC(float64_t, num_features);

		for (i=0; i<num_vectors; i++)
		{
		for (k=0; k<num_features; k++)
		sub_mean[k]=feature_matrix.matrix[i*num_features+k]-m_mean_vector.vector[k];

		cblas_dger(CblasColMajor,
			           num_features,num_features,
			           1.0,sub_mean,1,
			           sub_mean,1,
			           cov, num_features);
		}

		SG_FREE(sub_mean);

		for (i=0; i<num_features; i++)
		{
			for (j=0; j<num_features; j++)
				cov[i*num_features+j]/=(num_vectors-1);
		}

		SG_INFO("Computing Eigenvalues ... ")

		m_eigenvalues_vector.vector = SGMatrix<float64_t>::compute_eigenvectors(cov,num_features,num_features);
		m_eigenvalues_vector.vlen = num_features;
		num_dim=0;

		if (m_mode == FIXED_NUMBER)
		{
			ASSERT(m_target_dim <= num_features)
			num_dim = m_target_dim;
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

		SG_INFO("Done\nReducing from %i to %i features..", num_features, num_dim)

		m_transformation_matrix = SGMatrix<float64_t>(num_features,num_dim);
		num_old_dim = num_features;

		int32_t offs=0;
		for (i=num_features-num_dim; i<num_features; i++)
		{
			for (k=0; k<num_features; k++)
				if (m_whitening)
					m_transformation_matrix.matrix[offs+k*num_dim] =
						cov[num_features*i+k]/sqrt(m_eigenvalues_vector.vector[i]);
				else
					m_transformation_matrix.matrix[offs+k*num_dim] =
						cov[num_features*i+k];
			offs++;
		}

		SG_FREE(cov);
		m_initialized = true;
		return true;
	}

	return false;
}

void CPCA::cleanup()
{
	m_transformation_matrix=SGMatrix<float64_t>();
}

#ifdef HAVE_EIGEN3
SGMatrix<float64_t> CPCA::apply_to_feature_matrix(CFeatures* features)
{
	ASSERT(m_initialized)
	SGMatrix<float64_t> m = ((CDenseFeatures<float64_t>*) features)->get_feature_matrix();
	int32_t num_vectors = m.num_cols;
	int32_t num_features = m.num_rows;
	SG_INFO("get Feature matrix: %ix%i\n", num_vectors, num_features)

	MatrixXd final_feature_matrix;
	
	if (m.matrix)
	{
		SG_INFO("Preprocessing feature matrix\n")
		Map<MatrixXd> feature_matrix(m.matrix, num_features, num_vectors);
		VectorXd data_mean = feature_matrix.rowwise().sum()/(float64_t) num_vectors;
		MatrixXd feature_matrix_centered = feature_matrix.colwise()-data_mean;

		SG_INFO("Transforming feature matrix\n")
		Map<MatrixXd> transform_matrix(m_transformation_matrix.matrix, 
			m_transformation_matrix.num_rows, m_transformation_matrix.num_cols);
		final_feature_matrix = transform_matrix.transpose()*feature_matrix_centered;
	}

	SG_INFO("Copying eigen matrix to shogun matrix\n")	
	SGMatrix<float64_t> result_matrix = SGMatrix<float64_t>(num_dim, num_vectors);
	for (int32_t c=0; c<num_vectors; c++)
	{
		for (int32_t r=0; r<num_dim; r++)
			result_matrix.matrix[c*num_dim+r] = final_feature_matrix(r,c); 
	}  

	return result_matrix;
}
#endif //HAVE_EIGEN3

SGVector<float64_t> CPCA::apply_to_feature_vector(SGVector<float64_t> vector)
{
	float64_t* result = SG_MALLOC(float64_t, num_dim);
	float64_t* sub_mean = SG_MALLOC(float64_t, vector.vlen);

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
	return m_transformation_matrix;
}

SGVector<float64_t> CPCA::get_eigenvalues()
{
	return m_eigenvalues_vector;
}

SGVector<float64_t> CPCA::get_mean()
{
	return m_mean_vector;
}

#endif /* HAVE_LAPACK */
