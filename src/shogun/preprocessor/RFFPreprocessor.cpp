/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Evgeniy Andreev, Viktor Gal,
 *          Sergey Lisitsyn, Bjoern Esser, Sanuj Sharma, Saurabh Goyal
 */

#include <shogun/preprocessor/RFFPreprocessor.h>
#include <shogun/mathematics/RandomNamespace.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/NormalDistribution.h>
#include <shogun/kernel/GaussianKernel.h>

using namespace shogun;

RFFPreprocessor::RFFPreprocessor()
{
	SG_ADD(
	    &m_dim_output, "dim_output",
	    "Dimensionality of the output feature space.",
	    ParameterProperties::HYPER | ParameterProperties::CONSTRAIN,
	    SG_CONSTRAINT(positive<>()));
	SG_ADD(
	    (std::shared_ptr<Kernel>*)&m_kernel, "kernel", "Kernel object",
	    ParameterProperties::HYPER | ParameterProperties::CONSTRAIN,
		SG_CONSTRAINT(castable<ShiftInvariantKernel>()));
	SG_ADD(&m_basis, "basis", "Matrix of basis vectors");
	SG_ADD(&m_offset, "offset", "offset vector");
}

RFFPreprocessor::~RFFPreprocessor()
{
}

void RFFPreprocessor::init_basis(int32_t dim_input_space)
{
	io::info("Creating Fourier Basis Matrix {}x{}", m_dim_output, dim_input_space);

	m_basis = sample_spectral_density(dim_input_space);
	m_offset = SGVector<float64_t>(m_dim_output);

	UniformRealDistribution<float64_t> uniform(0.0, 2.0 * Math::PI);
	random::fill_array(m_offset, uniform, m_prng);
}

void RFFPreprocessor::fit_impl(const SGMatrix<float64_t>& feature_matrix)
{
	auto num_features = feature_matrix.num_cols;
	require(m_kernel, "Kernel not set");
	
	init_basis(num_features);
}

SGVector<float64_t> RFFPreprocessor::apply_to_feature_vector(SGVector<float64_t> vector)
{
	return static_cast<SGVector<float64_t>>(
	    apply_to_matrix(static_cast<SGMatrix<float64_t>>(vector)));
}

SGMatrix<float64_t> RFFPreprocessor::apply_to_matrix(SGMatrix<float64_t> matrix)
{
	auto num_vectors = matrix.num_cols;
	auto num_features = matrix.num_rows;

	assert_fitted();
	SG_DEBUG("Got Feature matrix: {}x{}", num_vectors, num_features);

	SGMatrix<float64_t> projection(m_dim_output, num_vectors);

	linalg::matrix_prod(m_basis,matrix,projection);
	linalg::add_vector(projection, m_offset, projection);
	linalg::cos(projection, projection);
	linalg::scale(projection, projection, std::sqrt(2.0 / m_dim_output));

	return projection;	
}

SGMatrix<float64_t> RFFPreprocessor::sample_spectral_density(int32_t dim_input_space) const
{

	SGMatrix<float64_t> sampled_kernel(m_dim_output, dim_input_space);

	if (auto kernel = std::dynamic_pointer_cast<GaussianKernel>(m_kernel))
	{
		NormalDistribution<float64_t> normal_dist;
		const auto width = kernel->get_width();
		const auto std_dev = std::sqrt(2.0 / width);
		std::generate(
		    sampled_kernel.begin(), sampled_kernel.end(),
		    [this, &normal_dist, &std_dev]() {
			    return std_dev * normal_dist(m_prng);
		    });
	}
	else
	{
		not_implemented(SOURCE_LOCATION);
	}
	
	return sampled_kernel;
}