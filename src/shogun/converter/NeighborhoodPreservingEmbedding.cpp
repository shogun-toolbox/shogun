/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Heiko Strathmann, Fernando Iglesias,
 *          Evan Shelhamer
 */

#include <shogun/converter/NeighborhoodPreservingEmbedding.h>
#include <shogun/lib/config.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>

using namespace shogun;

NeighborhoodPreservingEmbedding::NeighborhoodPreservingEmbedding() :
		LocallyLinearEmbedding()
{
}

NeighborhoodPreservingEmbedding::~NeighborhoodPreservingEmbedding()
{
}

const char* NeighborhoodPreservingEmbedding::get_name() const
{
	return "NeighborhoodPreservingEmbedding";
}

std::shared_ptr<Features>
NeighborhoodPreservingEmbedding::transform(std::shared_ptr<Features> features, bool inplace)
{
	auto dot_feats = std::static_pointer_cast<DotFeatures>(features);
	auto kernel = std::make_shared<LinearKernel>(dot_feats, dot_feats);
	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	parameters.n_neighbors = m_k;
	parameters.eigenshift = m_nullspace_shift;
	parameters.method = SHOGUN_NEIGHBORHOOD_PRESERVING_EMBEDDING;
	parameters.target_dimension = m_target_dim;
	parameters.kernel = kernel.get();
	parameters.features = dot_feats.get();
	return tapkee_embed(parameters);
}

