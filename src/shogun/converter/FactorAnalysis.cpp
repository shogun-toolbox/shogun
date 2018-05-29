/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Fernando Iglesias, Sergey Lisitsyn
 */

#include <shogun/converter/FactorAnalysis.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

CFactorAnalysis::CFactorAnalysis() :
		CEmbeddingConverter()
{
	// Sentinel value, it will be set appropriately if not modified by set_max_iteration
	m_max_iteration = 0;
	m_epsilon = 1e-5;
	init();
}

void CFactorAnalysis::init()
{
	SG_ADD(&m_max_iteration, "max_iteration", "maximum number of iterations", MS_NOT_AVAILABLE);
	SG_ADD(&m_epsilon, "epsilon", "convergence parameter", MS_NOT_AVAILABLE);
}

CFactorAnalysis::~CFactorAnalysis()
{
}

const char* CFactorAnalysis::get_name() const
{
	return "FactorAnalysis";
}

void CFactorAnalysis::set_max_iteration(const int32_t max_iteration)
{
	m_max_iteration = max_iteration;
}

int32_t CFactorAnalysis::get_max_iteration() const
{
	return m_max_iteration;
}

void CFactorAnalysis::set_epsilon(const float64_t epsilon)
{
	m_epsilon = epsilon;
}

float64_t CFactorAnalysis::get_epsilon() const
{
	return m_epsilon;
}

CFeatures* CFactorAnalysis::transform(CFeatures* features, bool inplace)
{
	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	parameters.max_iteration = m_max_iteration;
	parameters.features = (CDotFeatures*)features;
	parameters.fa_epsilon = m_epsilon;
	parameters.method = SHOGUN_FACTOR_ANALYSIS;
	parameters.target_dimension = m_target_dim;
	CDenseFeatures<float64_t>* embedding = tapkee_embed(parameters);
	return embedding;
}

