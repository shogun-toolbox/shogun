/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 1999-2008,2011 Soeren Sonnenburg
 * Written (W) 2014 Parijat Mazumdar
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2011 Berlin Institute of Technology
 */
#include <shogun/lib/config.h>

#if defined HAVE_EIGEN3
#include <shogun/preprocessor/PCA.h>
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

CPCA::CPCA(bool do_whitening, EPCAMode mode, float64_t thresh, EPCAMethod method, EPCAMemoryMode mem_mode)
: CDimensionReductionPreprocessor()
{
	init();
	m_whitening = do_whitening;
	m_mode = mode;
	m_thresh = thresh;
	m_mem_mode = mem_mode;
	m_method = method;
}

CPCA::CPCA(EPCAMethod method, bool do_whitening, EPCAMemoryMode mem_mode)
: CDimensionReductionPreprocessor()
{
	init();
	m_whitening = do_whitening;
	m_mem_mode = mem_mode;
	m_method = method;
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
	m_thresh = 1e-6;
	m_mem_mode = MEM_REALLOCATE;
	m_method = AUTO;	

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
	SG_ADD(&m_thresh, "m_thresh", "Cutoff threshold.", MS_AVAILABLE);
	SG_ADD((machine_int_t*) &m_mem_mode, "m_mem_mode", 
		"Memory mode (in-place or reallocation).", MS_NOT_AVAILABLE);
	SG_ADD((machine_int_t*) &m_method, "m_method", 
		"Method used for PCA calculation", MS_NOT_AVAILABLE);
}

CPCA::~CPCA()
{
}

bool CPCA::init(CFeatures* features)
{
	if (!m_initialized)
	{
		REQUIRE(features->get_feature_class()==C_DENSE, "PCA only works with dense features")
		REQUIRE(features->get_feature_type()==F_DREAL, "PCA only works with real features")

		SGMatrix<float64_t> feature_matrix = ((CDenseFeatures<float64_t>*)features)
									->get_feature_matrix();
		int32_t num_vectors = feature_matrix.num_cols;
		int32_t num_features = feature_matrix.num_rows;
		SG_INFO("num_examples: %ld num_features: %ld \n", num_vectors, num_features)

		// max target dim allowed
		int32_t max_dim_allowed = CMath::min(num_vectors, num_features);	
		num_dim=0;

		REQUIRE(m_target_dim<=max_dim_allowed,
			 "target dimension should be less or equal to than minimum of N and D")

		// center data
		Map<MatrixXd> fmatrix(feature_matrix.matrix, num_features, num_vectors);
		m_mean_vector = SGVector<float64_t>(num_features);
		Map<VectorXd> data_mean(m_mean_vector.vector, num_features);
 		data_mean = fmatrix.rowwise().sum()/(float64_t) num_vectors;
		fmatrix = fmatrix.colwise()-data_mean;

		m_eigenvalues_vector = SGVector<float64_t>(max_dim_allowed);
		Map<VectorXd> eigenValues(m_eigenvalues_vector.vector, max_dim_allowed);

		if (m_method == AUTO)
			m_method = (num_vectors>num_features) ? EVD : SVD;

		if (m_method == EVD)
		{
			// covariance matrix
			MatrixXd cov_mat(num_features, num_features);	
			cov_mat = fmatrix*fmatrix.transpose();
			cov_mat /= (num_vectors-1);

			SG_INFO("Computing Eigenvalues ... ")
			// eigen value computed
			SelfAdjointEigenSolver<MatrixXd> eigenSolve = 
					SelfAdjointEigenSolver<MatrixXd>(cov_mat);
			eigenValues = eigenSolve.eigenvalues().tail(max_dim_allowed);

			// target dimension
			switch (m_mode)
			{
				case FIXED_NUMBER :
					num_dim = m_target_dim;
					break;

				case VARIANCE_EXPLAINED :
					{
						float64_t eig_sum = eigenValues.sum();
						float64_t com_sum = 0;
						for (int32_t i=num_features-1; i<-1; i++)
						{
							num_dim++;
							com_sum += m_eigenvalues_vector.vector[i];
							if (com_sum/eig_sum>=m_thresh)
								break;
						}
					}
					break;

				case THRESHOLD :
					for (int32_t i=num_features-1; i<-1; i++)
					{
						if (m_eigenvalues_vector.vector[i]>m_thresh)
							num_dim++;
						else
							break;
					}
					break;
			};
			SG_INFO("Done\nReducing from %i to %i features..", num_features, num_dim)

			m_transformation_matrix = SGMatrix<float64_t>(num_features,num_dim);
			Map<MatrixXd> transformMatrix(m_transformation_matrix.matrix,
								 num_features, num_dim);
			num_old_dim = num_features;
                        
			// eigenvector matrix
			transformMatrix = eigenSolve.eigenvectors().block(0, 
						num_features-num_dim, num_features,num_dim);
			if (m_whitening)
			{
				for (int32_t i=0; i<num_dim; i++)
					transformMatrix.col(i) /= 
					sqrt(eigenValues[i+max_dim_allowed-num_dim]);
			}
		}

		else
		{
			// compute SVD of data matrix
			JacobiSVD<MatrixXd> svd(fmatrix.transpose(), ComputeThinU | ComputeThinV);

			// compute non-negative eigen values from singular values
			eigenValues = svd.singularValues();
			eigenValues = eigenValues.cwiseProduct(eigenValues)/(num_vectors-1);

			// target dimension
			switch (m_mode)
			{
				case FIXED_NUMBER :
					num_dim = m_target_dim;
					break;

				case VARIANCE_EXPLAINED :
					{
						float64_t eig_sum = eigenValues.sum();
						float64_t com_sum = 0;
						for (int32_t i=0; i<num_features; i++)
						{
							num_dim++;
							com_sum += m_eigenvalues_vector.vector[i];
							if (com_sum/eig_sum>=m_thresh)
								break;
						}
					}
					break;

				case THRESHOLD :
					for (int32_t i=0; i<num_features; i++)
					{
						if (m_eigenvalues_vector.vector[i]>m_thresh)
							num_dim++;
						else
							break;
					}
					break;
			};
			SG_INFO("Done\nReducing from %i to %i features..", num_features, num_dim)

			// right singular vectors form eigenvectors
			m_transformation_matrix = SGMatrix<float64_t>(num_features,num_dim);
			Map<MatrixXd> transformMatrix(m_transformation_matrix.matrix, num_features, num_dim);
			num_old_dim = num_features;
			transformMatrix = svd.matrixV().block(0, 0, num_features, num_dim);
			if (m_whitening)
			{
				for (int32_t i=0; i<num_dim; i++)
					transformMatrix.col(i) /= sqrt(eigenValues[i]);
			}
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

	if (m_mem_mode == MEM_IN_PLACE)
	{
		if (m.matrix)
		{
			SG_INFO("Preprocessing feature matrix\n")
			Map<MatrixXd> feature_matrix(m.matrix, num_features, num_vectors);
			VectorXd data_mean = feature_matrix.rowwise().sum()/(float64_t) num_vectors;
			feature_matrix = feature_matrix.colwise()-data_mean;

			feature_matrix.block(0,0,num_dim,num_vectors) = 
					transform_matrix.transpose()*feature_matrix;
	
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
	SGVector<float64_t> result = SGVector<float64_t>(num_dim);
	Map<VectorXd> resultVec(result.vector, num_dim);
	Map<VectorXd> inputVec(vector.vector, vector.vlen);

	Map<VectorXd> mean(m_mean_vector.vector, m_mean_vector.vlen);
	Map<MatrixXd> transformMat(m_transformation_matrix.matrix,
		 m_transformation_matrix.num_rows, m_transformation_matrix.num_cols);

	inputVec = inputVec-mean;
	resultVec = transformMat.transpose()*inputVec;
	inputVec = inputVec+mean;

	return result;
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
	return m_mem_mode;
}

void CPCA::set_memory_mode(EPCAMemoryMode e)
{
	m_mem_mode = e;
}

#endif // HAVE_EIGEN3
