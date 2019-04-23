/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Heiko Strathmann, Evan Shelhamer
 */

#include <shogun/converter/HessianLocallyLinearEmbedding.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>

using namespace shogun;

HessianLocallyLinearEmbedding::HessianLocallyLinearEmbedding() :
		LocallyLinearEmbedding()
{
}

HessianLocallyLinearEmbedding::~HessianLocallyLinearEmbedding()
{
}

const char* HessianLocallyLinearEmbedding::get_name() const
{
	return "HessianLocallyLinearEmbedding";
};

std::shared_ptr<Features>
HessianLocallyLinearEmbedding::transform(std::shared_ptr<Features> features, bool inplace)
{
	auto dot_feats = std::static_pointer_cast<DotFeatures>(features);
	std::shared_ptr<Kernel> kernel = std::make_shared<LinearKernel>(dot_feats, dot_feats);
	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	parameters.n_neighbors = m_k;
	parameters.eigenshift = m_nullspace_shift;
	parameters.method = SHOGUN_HESSIAN_LOCALLY_LINEAR_EMBEDDING;
	parameters.target_dimension = m_target_dim;
	parameters.kernel = kernel.get();
	return tapkee_embed(parameters);
}

