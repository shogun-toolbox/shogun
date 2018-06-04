/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Vladislav Horbatiuk, Bjoern Esser
 */

#include <shogun/converter/TDistributedStochasticNeighborEmbedding.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

CTDistributedStochasticNeighborEmbedding::CTDistributedStochasticNeighborEmbedding() :
		CEmbeddingConverter()
{
	// Default values
	m_perplexity = 30.0;
	m_theta = 0.5;
	init();
}

void CTDistributedStochasticNeighborEmbedding::init()
{
	SG_ADD(&m_perplexity, "perplexity", "perplexity", MS_NOT_AVAILABLE);
	SG_ADD(&m_theta, "theta", "learning rate", MS_NOT_AVAILABLE);
}

CTDistributedStochasticNeighborEmbedding::~CTDistributedStochasticNeighborEmbedding()
{
}

const char* CTDistributedStochasticNeighborEmbedding::get_name() const
{
	return "TDistributedStochasticNeighborEmbedding";
}

void CTDistributedStochasticNeighborEmbedding::set_theta(const float64_t theta)
{

	m_theta = theta;
}

float64_t CTDistributedStochasticNeighborEmbedding::get_theta() const
{
	return m_theta;
}

void CTDistributedStochasticNeighborEmbedding::set_perplexity(const float64_t perplexity)
{
	m_perplexity = perplexity;
}

float64_t CTDistributedStochasticNeighborEmbedding::get_perplexity() const
{
	return m_perplexity;
}

CFeatures* CTDistributedStochasticNeighborEmbedding::apply(
    CFeatures* features, bool inplace)
{
	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	parameters.sne_theta = m_theta;
	parameters.sne_perplexity = m_perplexity;
	parameters.features = (CDotFeatures*)features;

	parameters.method = SHOGUN_TDISTRIBUTED_STOCHASTIC_NEIGHBOR_EMBEDDING;
	parameters.target_dimension = m_target_dim;
	CDenseFeatures<float64_t>* embedding = tapkee_embed(parameters);
	return embedding;
}

