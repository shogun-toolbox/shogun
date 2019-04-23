/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Fernando Iglesias, Sergey Lisitsyn
 */

#include <shogun/converter/FactorAnalysis.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

FactorAnalysis::FactorAnalysis() :
		EmbeddingConverter()
{
	// Sentinel value, it will be set appropriately if not modified by set_max_iteration
	m_max_iteration = 0;
	m_epsilon = 1e-5;
	init();
}

void FactorAnalysis::init()
{
	SG_ADD(&m_max_iteration, "max_iteration", "maximum number of iterations");
	SG_ADD(&m_epsilon, "epsilon", "convergence parameter");
}

FactorAnalysis::~FactorAnalysis()
{
}

const char* FactorAnalysis::get_name() const
{
	return "FactorAnalysis";
}

void FactorAnalysis::set_max_iteration(const int32_t max_iteration)
{
	m_max_iteration = max_iteration;
}

int32_t FactorAnalysis::get_max_iteration() const
{
	return m_max_iteration;
}

void FactorAnalysis::set_epsilon(const float64_t epsilon)
{
	m_epsilon = epsilon;
}

float64_t FactorAnalysis::get_epsilon() const
{
	return m_epsilon;
}

std::shared_ptr<Features> FactorAnalysis::transform(std::shared_ptr<Features> features, bool inplace)
{
	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	parameters.max_iteration = m_max_iteration;
	parameters.features = (DotFeatures*)features.get();
	parameters.fa_epsilon = m_epsilon;
	parameters.method = SHOGUN_FACTOR_ANALYSIS;
	parameters.target_dimension = m_target_dim;
	return tapkee_embed(parameters);

}

