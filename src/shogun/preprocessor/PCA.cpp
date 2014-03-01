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
#include <shogun/lib/config.h>

#if defined(HAVE_LAPACK) && defined(HAVE_EIGEN3)
#include <shogun/preprocessor/PCA.h>
#include <shogun/mathematics/lapack.h>
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

CPCA::CPCA(bool do_whitening_, EPCAMode mode_, float64_t thresh_, EPCAMemoryMode mem_)
: CDimensionReductionPreprocessor()
{
	init();
	m_whitening = do_whitening_;
	m_mode = mode_;
	thresh = thresh_;
	mem_mode = mem_;
}

void CPCA::init()
{
	m_transformation_matrix = SGMatrix<float64_t>();
	m_mean_vector = SGVector<float64_t>();
	m_eigenvalues_vector = SGVector<float64_t>();
	num_dim = 0;
	m_initialized = false;
	m_whitening = false;
	m_mode = FIXED_NUMBER;
	thresh = 1e-6;
	mem_mode = MEM_REALLOCATE;

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
	SG_ADD((machine_int_t*) &mem_mode, "mem_mode", "Memory mode (in-place or reallocation).", MS_NOT_AVAILABLE);
}

CPCA::~CPCA()
{
}

bool CPCA::init(CFeatures* features)
{
	if (!m_initialized)
	{
		// loop variable
		int32_t i;

		ASSERT(features->get_feature_class()==C_DENSE)
		ASSERT(features->get_feature_type()==F_DREAL)

		int32_t num_vectors = ((CDenseFeatures<float64_t>*)features)->get_num_vectors();
		int32_t num_features = ((CDenseFeatures<float64_t>*)features)->get_num_features();
		SG_INFO("num_examples: %ld num_features: %ld \n", num_vectors, num_features)

		SGMatrix<float64_t> feature_matrix = ((CDenseFeatures<float64_t>*)features)->get_feature_matrix();
		m_mean_vector.vlen = num_features;
		m_mean_vector.vector = SG_CALLOC(float64_t, num_features);

		// center data
		Map<MatrixXd> fmatrix(feature_matrix.matrix, num_features, num_vectors);
		Map<VectorXd> data_mean(m_mean_vector.vector, num_features);
 		data_mean = fmatrix.rowwise().sum()/(float64_t) num_vectors;
		fmatrix = fmatrix.colwise()-data_mean;

		// covariance matrix
		MatrixXd cov_mat(num_features, num_features);
		cov_mat = fmatrix*fmatrix.transpose();
		cov_mat /= (num_vectors-1);

		SG_INFO("Computing Eigenvalues ... ")
		// eigen value computed
		SelfAdjointEigenSolver<MatrixXd> eigenSolve = SelfAdjointEigenSolver<MatrixXd>(cov_mat);
		m_eigenvalues_vector.vector = SG_MALLOC(float64_t, num_features);
		m_eigenvalues_vector.vlen = num_features;
		Map<VectorXd> eigenValues(m_eigenvalues_vector.vector, num_features);
		eigenValues = eigenSolve.eigenvalues();

		num_dim=0;
		if (m_mode == FIXED_NUMBER)
		{
			ASSERT(m_target_dim <= num_features)
			num_dim = m_target_dim;
		}
		if (m_mode == VARIANCE_EXPLAINED)
		{
			float64_t eig_sum = eigenValues.sum();
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
		Map<MatrixXd> transformMatrix(m_transformation_matrix.matrix, num_features, num_dim);
		num_old_dim = num_features;

		// eigenvector matrix
		transformMatrix = eigenSolve.eigenvectors().block(0, num_features-num_dim, num_features, num_dim);
		if (m_whitening)
		{
			for (i=0; i<num_dim; i++)
				transformMatrix.col(i) /= sqrt(eigenValues[i+num_features-num_dim]);
		}

		// restore feature matrix
		fmatrix = fmatrix.colwise()+data_mean;
		m_initialized = true;
		return true;
	}

	return false;
}

void CPCA::cleanup()
{
	m_transformation_matrix=SGMatrix<float64_t>();
        m_mean_vector = SGVector<float64_t>();
        m_eigenvalues_vector = SGVector<float64_t>();
	m_initialized = false;
}

SGMatrix<float64_t> CPCA::apply_to_feature_matrix(CFeatures* features)
{
	ASSERT(m_initialized)
	SGMatrix<float64_t> m = ((CDenseFeatures<float64_t>*) features)->get_feature_matrix();
	int32_t num_vectors = m.num_cols;
	int32_t num_features = m.num_rows;

	SG_INFO("Transforming feature matrix\n")
	Map<MatrixXd> transform_matrix(m_transformation_matrix.matrix,
			m_transformation_matrix.num_rows, m_transformation_matrix.num_cols);

	if (mem_mode == MEM_IN_PLACE)
	{
		if (m.matrix)
		{
			SG_INFO("Preprocessing feature matrix\n")
			Map<MatrixXd> feature_matrix(m.matrix, num_features, num_vectors);
			VectorXd data_mean = feature_matrix.rowwise().sum()/(float64_t) num_vectors;
			feature_matrix = feature_matrix.colwise()-data_mean;

			feature_matrix.block(0,0,num_dim,num_vectors) = transform_matrix.transpose()*feature_matrix;
	
			SG_INFO("Form matrix of target dimension")
			for (int32_t col=0; col<num_vectors; col++)
			{
				for (int32_t row=0; row<num_dim; row++)
					m.matrix[row*num_dim+col] = feature_matrix(row,col);
			}
			m.num_rows = num_dim;
			m.num_cols = num_vectors;
		}

		((CDenseFeatures<float64_t>*) features)->set_feature_matrix(m);
		return m;
	}
	else
	{
		SGMatrix<float64_t> ret(num_dim, num_vectors);
		Map<MatrixXd> ret_matrix(ret.matrix, num_dim, num_vectors);
		if (m.matrix)
		{
			SG_INFO("Preprocessing feature matrix\n")
			Map<MatrixXd> feature_matrix(m.matrix, num_features, num_vectors);
			VectorXd data_mean = feature_matrix.rowwise().sum()/(float64_t) num_vectors;
			feature_matrix = feature_matrix.colwise()-data_mean;

			ret_matrix = transform_matrix.transpose()*feature_matrix;
		}
		((CDenseFeatures<float64_t>*) features)->set_feature_matrix(ret);
		return ret;
	}
}

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

EPCAMemoryMode CPCA::get_memory_mode() const
{
	return mem_mode;
}

void CPCA::set_memory_mode(EPCAMemoryMode e)
{
	mem_mode = e;
}

#endif // HAVE_LAPACK && HAVE_EIGEN3
