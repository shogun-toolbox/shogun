/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soeren Sonnenburg, Heiko Strathmann,
 *          Fernando Iglesias, Viktor Gal, Evan Shelhamer
 */

#include <shogun/converter/KernelLocallyLinearEmbedding.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>
#include <utility>

using namespace shogun;

KernelLocallyLinearEmbedding::KernelLocallyLinearEmbedding() :
		LocallyLinearEmbedding()
{
}

KernelLocallyLinearEmbedding::KernelLocallyLinearEmbedding(std::shared_ptr<Kernel> kernel) :
		LocallyLinearEmbedding()
{
	set_kernel(std::move(kernel));
}

const char* KernelLocallyLinearEmbedding::get_name() const
{
	return "KernelLocallyLinearEmbedding";
};

KernelLocallyLinearEmbedding::~KernelLocallyLinearEmbedding()
{
}

std::shared_ptr<Features>
KernelLocallyLinearEmbedding::transform(std::shared_ptr<Features> features, bool inplace)
{
	ASSERT(features)


	// get dimensionality and number of vectors of data
	int32_t N = features->get_num_vectors();
	if (m_k>=N)
		error("Number of neighbors ({}) should be less than number of objects ({}).",
		         m_k, N);

	// compute kernel matrix
	ASSERT(m_kernel)
	m_kernel->init(features,features);
	auto embedding = embed_kernel(m_kernel);
	m_kernel->cleanup();

	return embedding;
}

std::shared_ptr<DenseFeatures<float64_t>> KernelLocallyLinearEmbedding::embed_kernel(const std::shared_ptr<Kernel>& kernel)
{
	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	parameters.n_neighbors = m_k;
	parameters.eigenshift = m_nullspace_shift;
	parameters.method = SHOGUN_KERNEL_LOCALLY_LINEAR_EMBEDDING;
	parameters.target_dimension = m_target_dim;
	parameters.kernel = kernel.get();
	return tapkee_embed(parameters);
}

