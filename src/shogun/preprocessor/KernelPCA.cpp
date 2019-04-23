/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Michele Mazzoni, Evan Shelhamer,
 *          Heiko Strathmann, Evgeniy Andreev, Thoralf Klein, Giovanni De Toni
 */

#include <shogun/preprocessor/KernelPCA.h>
#include <shogun/lib/config.h>
#include <shogun/mathematics/Math.h>

#include <limits>
#include <string.h>
#include <stdlib.h>

#include <shogun/features/Features.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

using namespace shogun;

KernelPCA::KernelPCA() : Preprocessor()
{
	init();
}

KernelPCA::KernelPCA(std::shared_ptr<Kernel> k) : Preprocessor()
{
	init();
	set_kernel(k);
}

void KernelPCA::init()
{
	m_fitted = false;
	m_init_features = NULL;
	m_transformation_matrix = SGMatrix<float64_t>();
	m_bias_vector = SGVector<float64_t>();
	m_target_dim = 1;
	m_kernel = NULL;

	SG_ADD(&m_transformation_matrix, "transformation_matrix",
		"matrix used to transform data");
	SG_ADD(&m_bias_vector, "bias_vector",
		"bias vector used to transform data");
	SG_ADD(
	    &m_target_dim, "target_dim", "target dimensionality of preprocessor",
	    ParameterProperties::HYPER);
	SG_ADD(&m_kernel, "kernel", "kernel to be used", ParameterProperties::HYPER);
}

void KernelPCA::cleanup()
{
	m_transformation_matrix = SGMatrix<float64_t>();
	m_bias_vector = SGVector<float64_t>();

	m_fitted = false;
}

KernelPCA::~KernelPCA()
{

}

void KernelPCA::fit(std::shared_ptr<Features> features)
{
	REQUIRE(m_kernel, "Kernel not set\n");

	if (m_fitted)
		cleanup();


	m_init_features = features;

	m_kernel->init(features, features);
	SGMatrix<float64_t> kernel_matrix = m_kernel->get_kernel_matrix();
	m_kernel->cleanup();
	int32_t n = kernel_matrix.num_cols;
	int32_t m = kernel_matrix.num_rows;
	ASSERT(n == m)
	if (m_target_dim > n)
	{
		SG_SWARNING(
		    "Target dimension (%d) is not a valid value, it must be"
		    "less or equal than the number of vectors."
		    "Setting it to maximum allowed size (%d).",
		    m_target_dim, n);
		m_target_dim = n;
	}

	auto bias_tmp = linalg::rowwise_sum(kernel_matrix);
	linalg::scale(bias_tmp, bias_tmp, -1.0 / n);
	auto s = linalg::sum(bias_tmp) / n;
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
		// normalize and trap divide by zero and negative eigenvalues
		auto idx = m_target_dim - i - 1;
		auto vec = eigenvectors.get_column(idx);
		linalg::scale(
		    vec, vec, 1.0 / std::sqrt(std::max(std::numeric_limits<float64_t>::epsilon(), eigenvalues[idx])));
		m_transformation_matrix.set_column(i, vec);
	}

	m_bias_vector = SGVector<float64_t>(m_target_dim);
	linalg::matrix_prod(m_transformation_matrix, bias_tmp, m_bias_vector, true);

	m_fitted = true;
	SG_INFO("Done\n")
}

std::shared_ptr<Features> KernelPCA::transform(std::shared_ptr<Features> features, bool inplace)
{
	assert_fitted();

	if (std::dynamic_pointer_cast<DenseFeatures<float64_t>>(features))
	{
		auto feature_matrix = apply_to_feature_matrix(features);
		return std::make_shared<DenseFeatures<float64_t>>(feature_matrix);
	}

	if (features->get_feature_class() == C_STRING)
	{
		return apply_to_string_features(features);
	}

	SG_ERROR("Feature type %d not supported\n", features->get_feature_type());
	return NULL;
}

SGMatrix<float64_t> KernelPCA::apply_to_feature_matrix(std::shared_ptr<Features> features)
{
	assert_fitted();
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

SGVector<float64_t> KernelPCA::apply_to_feature_vector(SGVector<float64_t> vector)
{
	assert_fitted();

	auto features =
	    std::make_shared<DenseFeatures<float64_t>>(SGMatrix<float64_t>(vector));


	SGMatrix<float64_t> result_matrix = apply_to_feature_matrix(features);


	return SGVector<float64_t>(result_matrix);
}

std::shared_ptr<DenseFeatures<float64_t>> KernelPCA::apply_to_string_features(std::shared_ptr<Features> features)
{
	assert_fitted();

	int32_t num_vectors = features->get_num_vectors();
	int32_t i,j,k;
	int32_t n = m_transformation_matrix.num_cols;

	m_kernel->init(features,m_init_features);

	SGMatrix<float64_t> new_feature_matrix(m_target_dim * num_vectors);

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

	return std::make_shared<DenseFeatures<float64_t>>(SGMatrix<float64_t>(new_feature_matrix,m_target_dim,num_vectors));
}

EFeatureClass KernelPCA::get_feature_class()
{
	return C_ANY;
}

EFeatureType KernelPCA::get_feature_type()
{
	return F_ANY;
}

void KernelPCA::set_target_dim(int32_t dim)
{
	ASSERT(dim > 0)
	m_target_dim = dim;
}

int32_t KernelPCA::get_target_dim() const
{
	return m_target_dim;
}

void KernelPCA::set_kernel(std::shared_ptr<Kernel> kernel)
{


	m_kernel = kernel;
}

std::shared_ptr<Kernel> KernelPCA::get_kernel() const
{

	return m_kernel;
}
