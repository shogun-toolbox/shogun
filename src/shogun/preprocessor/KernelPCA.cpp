/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Michele Mazzoni, Evan Shelhamer, 
 *          Heiko Strathmann, Evgeniy Andreev, Thoralf Klein, Giovanni De Toni
 */

#include <shogun/preprocessor/KernelPCA.h>
#include <shogun/lib/config.h>
#include <shogun/mathematics/Math.h>

#include <string.h>
#include <stdlib.h>

#include <shogun/features/Features.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/preprocessor/DimensionReductionPreprocessor.h>

using namespace shogun;

CKernelPCA::CKernelPCA() : CDimensionReductionPreprocessor()
{
	init();
}

CKernelPCA::CKernelPCA(CKernel* k) : CDimensionReductionPreprocessor()
{
	init();
	set_kernel(k);
}

void CKernelPCA::init()
{
	m_initialized = false;
	m_init_features = NULL;
	m_transformation_matrix = SGMatrix<float64_t>();
	m_bias_vector = SGVector<float64_t>();

	SG_ADD(&m_transformation_matrix, "transformation_matrix",
		"matrix used to transform data", MS_NOT_AVAILABLE);
	SG_ADD(&m_bias_vector, "bias_vector",
		"bias vector used to transform data", MS_NOT_AVAILABLE);
}

void CKernelPCA::cleanup()
{
	m_transformation_matrix = SGMatrix<float64_t>();
	m_bias_vector = SGVector<float64_t>();

	if (m_init_features)
		SG_UNREF(m_init_features);

	m_initialized = false;
}

CKernelPCA::~CKernelPCA()
{
	if (m_init_features)
		SG_UNREF(m_init_features);
}

bool CKernelPCA::init(CFeatures* features)
{
	if (!m_initialized && m_kernel)
	{
		SG_REF(features);
		m_init_features = features;

		m_kernel->init(features,features);
		SGMatrix<float64_t> kernel_matrix = m_kernel->get_kernel_matrix();
		m_kernel->cleanup();
		int32_t n = kernel_matrix.num_cols;
		int32_t m = kernel_matrix.num_rows;
		ASSERT(n==m)
		if (m_target_dim > n)
		{
			SG_SWARNING(
			    "Target dimension (%d) is not a valid value, it must be"
			    "less or equal than the number of vectors."
			    "Setting it to maximum allowed size (%d).",
			    m_target_dim, n);
			m_target_dim = n;
		}

		SGVector<float64_t> bias_tmp = linalg::rowwise_sum(kernel_matrix);
		linalg::scale(bias_tmp, bias_tmp, -1.0 / n);
		float64_t s = linalg::sum(bias_tmp) / n;
		linalg::add_scalar(bias_tmp, -s);

		linalg::center_matrix(kernel_matrix);

		SGVector<float64_t> eigenvalues(m_target_dim);
		SGMatrix<float64_t> eigenvectors(kernel_matrix.num_rows, m_target_dim);
		linalg::eigen_solver_symmetric(
		    kernel_matrix, eigenvalues, eigenvectors, m_target_dim);

		m_transformation_matrix =
		    SGMatrix<float64_t>(kernel_matrix.num_rows, m_target_dim);
		// eigenvalues are in increasing order
		for (int32_t i = 0; i < m_target_dim; i++)
		{
			//normalize and trap divide by zero and negative eigenvalues
			auto idx = m_target_dim - i - 1;
			auto vec = eigenvectors.get_column(idx);
			linalg::scale(
			    vec, vec,
			    1.0 / std::sqrt(CMath::max(1e-16, eigenvalues[idx])));
			m_transformation_matrix.set_column(i, vec);
		}

		m_bias_vector = SGVector<float64_t>(m_target_dim);
		linalg::matrix_prod(
		    m_transformation_matrix, bias_tmp, m_bias_vector, true);

		m_initialized=true;
		SG_INFO("Done\n")
		return true;
	}
	return false;
}

SGMatrix<float64_t> CKernelPCA::apply_to_feature_matrix(CFeatures* features)
{
	ASSERT(m_initialized)
	int32_t n = m_init_features->get_num_vectors();

	m_kernel->init(features, m_init_features);
	auto kernel_matrix = m_kernel->get_kernel_matrix();

	auto rows_sum = linalg::rowwise_sum(kernel_matrix);
	linalg::add_vector(kernel_matrix, rows_sum, kernel_matrix, 1.0, -1.0 / n);

	SGMatrix<float64_t> new_feature_matrix =
	    linalg::matrix_prod(m_transformation_matrix, kernel_matrix, true, true);

	linalg::add_vector(new_feature_matrix, m_bias_vector, new_feature_matrix);

	m_kernel->cleanup();
	return new_feature_matrix;
}

SGVector<float64_t> CKernelPCA::apply_to_feature_vector(SGVector<float64_t> vector)
{
	ASSERT(m_initialized)

	CFeatures* features =
	    new CDenseFeatures<float64_t>(SGMatrix<float64_t>(vector));
	SG_REF(features)

	SGMatrix<float64_t> result_matrix = apply_to_feature_matrix(features);

	SG_UNREF(features)
	return SGVector<float64_t>(result_matrix);
}

CDenseFeatures<float64_t>* CKernelPCA::apply_to_string_features(CFeatures* features)
{
	ASSERT(m_initialized)

	int32_t num_vectors = features->get_num_vectors();
	int32_t i,j,k;
	int32_t n = m_transformation_matrix.num_cols;

	m_kernel->init(features,m_init_features);

	float64_t* new_feature_matrix = SG_MALLOC(float64_t, m_target_dim*num_vectors);

	for (i=0; i<num_vectors; i++)
	{
		for (j=0; j<m_target_dim; j++)
			new_feature_matrix[i*m_target_dim+j] = m_bias_vector.vector[j];

		for (j=0; j<n; j++)
		{
			float64_t kij = m_kernel->kernel(i,j);

			for (k=0; k<m_target_dim; k++)
				new_feature_matrix[k+i*m_target_dim] += kij*m_transformation_matrix.matrix[(n-k-1)*n+j];
		}
	}

	m_kernel->cleanup();

	return new CDenseFeatures<float64_t>(SGMatrix<float64_t>(new_feature_matrix,m_target_dim,num_vectors));
}
