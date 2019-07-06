/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Bjoern Esser
 */

#include <shogun/ensemble/WeightedMajorityVote.h>
#include <shogun/base/Parameter.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>
#include <map>

using namespace shogun;

WeightedMajorityVote::WeightedMajorityVote()
	: CombinationRule()
{
	init();
	register_parameters();
}

WeightedMajorityVote::WeightedMajorityVote(SGVector<float64_t>& weights)
	: CombinationRule()
{
	init();
	register_parameters();
	m_weights = weights;
}

WeightedMajorityVote::~WeightedMajorityVote()
{

}

SGVector<float64_t> WeightedMajorityVote::combine(const SGMatrix<float64_t>& ensemble_result) const
{
	require(m_weights.vlen == ensemble_result.num_cols, "The number of results and weights does not match!");
	SGVector<float64_t> mv(ensemble_result.num_rows);
	for (index_t i = 0; i < ensemble_result.num_rows; ++i)
	{
		SGVector<float64_t> rv = ensemble_result.get_row_vector(i);
		mv[i] = combine(rv);
	}

	return mv;
}

float64_t WeightedMajorityVote::combine(const SGVector<float64_t>& ensemble_result) const
{
	return weighted_combine(ensemble_result);
}

float64_t WeightedMajorityVote::weighted_combine(const SGVector<float64_t>& ensemble_result) const
{
	require(m_weights.vlen == ensemble_result.vlen, "The number of results and weights does not match!");
	std::map<index_t, float64_t> freq;
	std::map<index_t, float64_t>::iterator it;
	index_t max_label = -100;
	float64_t max = Math::ALMOST_NEG_INFTY;

	for (index_t i = 0; i < ensemble_result.vlen; ++i)
	{
		if (Math::is_nan(ensemble_result[i]))
			continue;

		it = freq.find(ensemble_result[i]);
		if (it == freq.end())
		{
			freq.insert(std::make_pair(ensemble_result[i], m_weights[i]));
			if (max < m_weights[i])
			{
				max_label = ensemble_result[i];
				max = m_weights[i];
			}
		}
		else
		{
			it->second += m_weights[i];
			if (max < it->second)
			{
				max_label = it->first;
				max = it->second;
			}
		}
	}

	return max_label;
}

void WeightedMajorityVote::set_weights(SGVector<float64_t>& w)
{
	m_weights = w;
}

SGVector<float64_t> WeightedMajorityVote::get_weights() const
{
	return m_weights;
}

void WeightedMajorityVote::init()
{
	m_weights = SGVector<float64_t>();
}

void WeightedMajorityVote::register_parameters()
{
	SG_ADD(&m_weights, "weights", "Weights for the majority vote", ParameterProperties::HYPER);
}
