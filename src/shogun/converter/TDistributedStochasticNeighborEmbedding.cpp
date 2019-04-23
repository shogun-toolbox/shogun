/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Vladislav Horbatiuk, Bjoern Esser
 */

#include <shogun/converter/TDistributedStochasticNeighborEmbedding.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

TDistributedStochasticNeighborEmbedding::TDistributedStochasticNeighborEmbedding() :
		EmbeddingConverter()
{
	// Default values
	m_perplexity = 30.0;
	m_theta = 0.5;
	init();
}

void TDistributedStochasticNeighborEmbedding::init()
{
	SG_ADD(&m_perplexity, "perplexity", "perplexity");
	SG_ADD(&m_theta, "theta", "learning rate");
}

TDistributedStochasticNeighborEmbedding::~TDistributedStochasticNeighborEmbedding()
{
}

const char* TDistributedStochasticNeighborEmbedding::get_name() const
{
	return "TDistributedStochasticNeighborEmbedding";
}

void TDistributedStochasticNeighborEmbedding::set_theta(const float64_t theta)
{

	m_theta = theta;
}

float64_t TDistributedStochasticNeighborEmbedding::get_theta() const
{
	return m_theta;
}

void TDistributedStochasticNeighborEmbedding::set_perplexity(const float64_t perplexity)
{
	m_perplexity = perplexity;
}

float64_t TDistributedStochasticNeighborEmbedding::get_perplexity() const
{
	return m_perplexity;
}

std::shared_ptr<Features> TDistributedStochasticNeighborEmbedding::transform(
    std::shared_ptr<Features> features, bool inplace)
{
	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	parameters.sne_theta = m_theta;
	parameters.sne_perplexity = m_perplexity;
	parameters.features = (DotFeatures*)features.get();
	parameters.method = SHOGUN_TDISTRIBUTED_STOCHASTIC_NEIGHBOR_EMBEDDING;
	parameters.target_dimension = m_target_dim;
	return tapkee_embed(parameters);
}

